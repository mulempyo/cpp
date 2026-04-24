#include <rclcpp/rclcpp.hpp>
#include "slam_algorithm.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/types/slam2d/types_slam2d.h>  
#include <g2o/core/robust_kernel_impl.h>
#include <pcl/registration/icp.h>
#include <iostream>
#include <fstream>
#include <robust_kernel_io.h>
#include <cuda_runtime.h>

G2O_USE_OPTIMIZATION_LIBRARY(csparse)

Eigen::Matrix4f runICPCUDA(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& src_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt_cloud,
    int max_iterations,
    Eigen::Matrix4f init_guess);

namespace graph_slam {

    GraphSLAM::GraphSLAM(const std::string& solver_type) {
        graph = std::make_shared<g2o::SparseOptimizer>();
        
        std::cout << "construct solver: " << solver_type << std::endl;
        g2o::OptimizationAlgorithmFactory* solver_factory = g2o::OptimizationAlgorithmFactory::instance();
        g2o::OptimizationAlgorithmProperty solver_property;
        g2o::OptimizationAlgorithm* solver = solver_factory->construct(solver_type, solver_property);
        graph->setAlgorithm(solver);

        if(!graph->solver()) {
            std::cerr << "error: failed to allocate solver!" << std::endl;
            solver_factory->listSolvers(std::cerr);
            return;
        }
        std::cout << "done" << std::endl;
    }

    int GraphSLAM::num_vertices() const {
        return graph->vertices().size();
    }

    int GraphSLAM::num_edges() const {
        return graph->edges().size();
    }

    g2o::VertexSE2* GraphSLAM::add_se2_node(const Eigen::Vector3d& pose) { 
        g2o::VertexSE2* vertex(new g2o::VertexSE2());
        static int vertex_counter = 0;
        vertex->setId(vertex_counter++);
        vertex->setEstimate(g2o::SE2(pose[0], pose[1], pose[2])); // x, y, theta

        if (!graph->addVertex(vertex)) {
            std::cerr << "Failed to add vertex to graph!" << std::endl;
            delete vertex;
            return nullptr;
        }else{
            v = graph->vertices().size();
        }
        return vertex;
    }

    g2o::EdgeSE2* GraphSLAM::add_se2_edge(g2o::VertexSE2* v1, g2o::VertexSE2* v2, const Eigen::Vector3d& relative_pose, const Eigen::Matrix3d& information_matrix) {
        static int edge_counter = 0; 
        g2o::EdgeSE2* edge = new g2o::EdgeSE2();

        edge->vertices()[0] = v1;
        edge->vertices()[1] = v2;
        edge->setMeasurement(g2o::SE2(relative_pose[0], relative_pose[1], relative_pose[2]));
        edge->setInformation(information_matrix);

        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(1.0);                
        edge->setRobustKernel(rk);

        edge->setId(edge_counter++);
        if (!graph->addEdge(edge)) {
            std::cerr << "Failed to add edge to graph!" << std::endl;
            delete edge;
            return nullptr;
        }else {
            e = graph->edges().size();
        }

        return edge;

    }

    void GraphSLAM::optimize(int num_iterations) {

        if (!graph) {
            std::cerr << "Error: graph is nullptr" << std::endl;
            return;
        }
    
        if (e < 10) {
            std::cerr << "Not enough edges for optimization" << std::endl;
            return;
        }

        if (v < 10) {
            std::cerr << "Not enough vertexs for optimization" << std::endl;
            return;
        }

        bool optimize = graph->initializeOptimization();
        if (!optimize) {
           std::cerr << "Error: initializeOptimization() failed!" << std::endl;
           return;
        }

        graph->setVerbose(false);
        int result = graph->optimize(num_iterations);


        if (result <= 0) {
            std::cerr << "Optimization failed" << std::endl;
        }
    }

    Eigen::Vector3d GraphSLAM::getOptimizedPose() {
        if (graph->vertices().empty()) {
            std::cerr << "[GraphSLAM] No vertices in graph, returning default pose." << std::endl;
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        }
    
        int last_id = graph->vertices().size() - 1;

         g2o::VertexSE2* last_vertex = dynamic_cast<g2o::VertexSE2*>(graph->vertex(last_id));  
        if (!last_vertex) {
            std::cerr << "[GraphSLAM] last_vertex is null, returning default pose." << std::endl;
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        }
    
        g2o::SE2 optimized_pose = last_vertex->estimate();
        return Eigen::Vector3d(optimized_pose.translation()[0], optimized_pose.translation()[1], optimized_pose.rotation().angle());
    }

    Eigen::Matrix3d GraphSLAM::makeSE2(double x, double y, double th){
        Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
        double c = std::cos(th), s = std::sin(th);
        T(0,0)=c; T(0,1)=-s;
        T(1,0)=s; T(1,1)= c;
        T(0,2)=x; T(1,2)=y;
        return T;
    }
      
      Eigen::Vector3d GraphSLAM::compute_scan_matching(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& previous_scan,
    Eigen::Vector3d current_pose, Eigen::Vector3d previous_pose)
{
    if (current_scan->empty() || previous_scan->empty()) {
        std::cout << "[CUDA ICP] One of the scans is empty!" << std::endl;
        return Eigen::Vector3d::Zero();
    }

    static double odom_dist = 0.0;

    // odom relative motion in previous local(base) frame
    double odom_dx = current_pose.x() - previous_pose.x();
    double odom_dy = current_pose.y() - previous_pose.y();
    double odom_dtheta = normalizeAngle(current_pose.z() - previous_pose.z());

    double yaw1 = previous_pose.z();

    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    double dx_local =  std::cos(yaw1) * odom_dx + std::sin(yaw1) * odom_dy;
    double dy_local = -std::sin(yaw1) * odom_dx + std::cos(yaw1) * odom_dy;

    std::cout << "dx_local: " << dx_local
          << " dy_local: " << dy_local
          << " odom_dtheta: " << odom_dtheta << std::endl;

    init_guess(0,0) = std::cos(odom_dtheta);
    init_guess(0,1) = -std::sin(odom_dtheta);
    init_guess(1,0) = std::sin(odom_dtheta);
    init_guess(1,1) = std::cos(odom_dtheta);
    init_guess(0,3) = dx_local;
    init_guess(1,3) = dy_local;
    
    Eigen::Matrix4f T = runICPCUDA(current_scan, previous_scan, 20, init_guess);

    double icp_dx = T(0, 3);
    double icp_dy = T(1, 3);
    double icp_theta = std::atan2(T(1,0), T(0,0));

    std::cout << "icp_dx: " << icp_dx
          << " icp_dy: " << icp_dy
          << " icp_theta: " << icp_theta << std::endl;

    double odom_step = std::hypot(dx_local, dy_local);      

    if(std::abs(odom_dx) < 0.00001 && std::abs(odom_dy) < 0.0001 && std::abs(odom_dtheta) < 0.001){
       icp_dx = 0.0;
       icp_dy = 0.0;
       icp_theta = 0.0;
       return Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    double dx_error = icp_dx - odom_dx;
    double dy_error = icp_dy - odom_dy;
    double dtheta_error = normalizeAngle(icp_theta - odom_dtheta);

    std::cout << "dx_error: " << dx_error
          << " dy_error: " << dy_error
          << " dtheta_error: " << dtheta_error << std::endl;

    bool good_icp = std::abs(dx_error) < 0.01 && std::abs(dy_error) < 0.01 && std::abs(dtheta_error) < 0.02;

    double fused_dx = dx_local;
    double fused_dy = dy_local;
    double fused_dtheta = odom_dtheta;
 
    if(good_icp && !(odom_step < 0.003 && std::abs(odom_dtheta) < 0.003)){
        double alpha_pos = 0.3;
        double alpha_yaw = 0.4;
        fused_dx = (1.0 - alpha_pos) * dx_local + alpha_pos * icp_dx;
        fused_dy = (1.0 - alpha_pos) * dy_local + alpha_pos * icp_dy;
        fused_dtheta = normalizeAngle((1.0 - alpha_yaw) * odom_dtheta + alpha_yaw * icp_theta);
    }

    fused_dist += std::hypot(fused_dx, fused_dy);
    odom_dist += std::hypot(dx_local, dy_local);

    std::cout << "fused dist: " << fused_dist << std::endl;
    std::cout << "odom_dist: " << odom_dist << std::endl;

    return Eigen::Vector3d(
        fused_dx,
        fused_dy,
        fused_dtheta 
    );
}

/*
src 점을 하나씩 가져옴
ICP 변환 적용
tgt에서 가장 가까운 점 찾기
대응점인지 판단
평균 거리 계산
inlier ratio 계산
평균 오차 계산
작을수록 좋다.
*/

double GraphSLAM::computeMeanError( // 공간적 일관성 검사: 공간적으로 정합이 잘 되는지 평가
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt,
    const Eigen::Matrix4f& T,
    double max_corr_dist,
    double& inlier_ratio)
{
    int valid = 0;
    double sum = 0.0;

    for (size_t i = 0; i < src->size(); ++i) {
        Eigen::Vector4f p(src->points[i].x, src->points[i].y, src->points[i].z, 1.0f);
        Eigen::Vector4f tp = T * p;

        double best = 1e9;
        for (size_t j = 0; j < tgt->size(); ++j) {
            double dx = tp[0] - tgt->points[j].x;
            double dy = tp[1] - tgt->points[j].y;
            double d = std::sqrt(dx*dx + dy*dy);
            if (d < best) best = d;
        }

        if (best < max_corr_dist) {
            sum += best;
            valid++;
        }
    }

    inlier_ratio = src->empty() ? 0.0 : static_cast<double>(valid) / src->size();
    if (valid == 0) return 1e9;
    return sum / valid;
}

double GraphSLAM::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// 최근 스캔 경향성 기반 이상치 제거: 최근 연속 스캔은 신뢰한다고 가정하고 이상치를 제거하는 시간적 일관성 검사
graph_slam::GraphSLAM::MotionStats GraphSLAM::computeRecentMotionStats(
    std::shared_ptr<graph_slam::GraphSLAM> slam,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& past_scans,
    int recent_window, Eigen::Vector3d current_pose, Eigen::Vector3d previous_pose)
{
    MotionStats stats;

    std::vector<double> trans_list;
    std::vector<double> rot_list;

    int start_idx = std::max(1, static_cast<int>(past_scans.size()) - recent_window); // 최근 recent_window개의 연속 스캔 쌍

    for (int i = start_idx; i < static_cast<int>(past_scans.size()); ++i) {
        Eigen::Vector3d rel = slam->compute_scan_matching(past_scans[i], past_scans[i - 1], current_pose, previous_pose); // compute_scan_matching(past_scans[i], past_scans[i - 1])로 상대 이동량을 구하고

        double trans = rel.head<2>().norm();
        double rot = std::fabs(normalizeAngle(rel[2]));

        trans_list.push_back(trans);
        rot_list.push_back(rot);
    }

    if (trans_list.size() < 2 || rot_list.size() < 2) {
        return stats;
    } 
    // 이동의 평균(mean) 과 표준편차(stddev) 를 계산 후 이상치 제거
    auto filterOutliers = [this](const std::vector<double>& data, double& mean, double& stddev, double sigma_scale = 2.5){
        std::vector<double> filtered;

        if (data.size() < 2) return data;

        mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        //std::cerr << "mean: " << mean << std::endl;

        double var = 0.0;
        for (double v : data) {
            double d = v - mean;
            var += d * d;
        }
        var /= data.size();
        stddev = std::sqrt(var);
        //std::cerr << "stddev:" << stddev << std::endl;

        double thresh = mean + sigma_scale * stddev;
        //std::cerr << "thresh: " << thresh << std::endl;
        for (double v : data) {
            if (v <= thresh) {
                filtered.push_back(v);
            }
        }

        return filtered;
    };

    trans_list = filterOutliers(trans_list, stats.mean_trans, stats.std_trans);
    rot_list   = filterOutliers(rot_list, stats.mean_rot, stats.std_rot);
    // MotionStats로 반환함
    stats.valid = true;
    return stats;
}
 
    void GraphSLAM::detect_loop_closure(
  std::shared_ptr<graph_slam::GraphSLAM> slam,
  const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& past_scans,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan,
  Eigen::Vector3d current_pose, Eigen::Vector3d previous_pose)
{
  const int curr_id = static_cast<int>(past_scans.size()) - 1;

  // std::cerr << "step 1" << std::endl;
  // std::cerr << "past_scans_.size(): " << past_scans.size() << std::endl;
  MotionStats stats = computeRecentMotionStats(slam, past_scans, 10, current_pose, previous_pose);
    // std::cerr << "step 2" << std::endl;
    if (!stats.valid) {
        return;
    }

    int best_id = -1;
    Eigen::Vector3d best_relative_pose = Eigen::Vector3d::Zero();
    double best_score = std::numeric_limits<double>::max();
    double best_inlier_ratio = 0.0;
    double best_mean_error = std::numeric_limits<double>::max();

    double odom_x = current_pose.x() - previous_pose.x();
    double odom_y = current_pose.y() - previous_pose.y();
    double odom_theta = std::atan2(
            std::sin(current_pose.z() - previous_pose.z()),
            std::cos(current_pose.z() - previous_pose.z())
        );

    double prev_yaw = previous_pose.z();

    double dx_local =  std::cos(prev_yaw) * odom_x + std::sin(prev_yaw) * odom_y;
    double dy_local = -std::sin(prev_yaw) * odom_x + std::cos(prev_yaw) * odom_y;

    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    init_guess(0,0) = std::cos(odom_theta);
    init_guess(0,1) = -std::sin(odom_theta);
    init_guess(1,0) = std::sin(odom_theta);
    init_guess(1,1) = std::cos(odom_theta);
    init_guess(0,3) = dx_local;
    init_guess(1,3) = dy_local;

  for (size_t i = 0; i < past_scans.size(); ++i) {
    // std::cerr << "step 3" << std::endl;
        int candidate_id = static_cast<int>(i);
        // 최근 프레임 제외
        if (std::abs(curr_id - static_cast<int>(i)) < 20) {
            continue;
        }

        Eigen::Matrix4f T = runICPCUDA(current_scan, past_scans[i], 20, init_guess);

        // 공간적 일관성 검사 후 이상치 제거
        // mean_error가 작을수록 두 포인트클라우드가 더 잘 맞는것을 의미한다.
        double inlier_ratio = 0.0;
        double mean_error = computeMeanError(current_scan, past_scans[i], T, 0.15, inlier_ratio); 
        // std::cerr << "inlier_ratio: " << inlier_ratio << std::endl;
        // std::cerr << "step 4" << std::endl;
        if (mean_error > 0.4 || inlier_ratio < 0.7){
            return;
        }

        double score = mean_error; // 제일 작은 mean_error일때 그때의 id와 포즈를 사용한다.

        if (score < best_score) {
            best_score = score;
            best_id = candidate_id;
            best_mean_error = mean_error;
            best_inlier_ratio = inlier_ratio;

            double dx = T(0, 3);
            double dy = T(1, 3);
            double yaw = std::atan2(T(1, 0), T(0, 0));
            best_relative_pose = Eigen::Vector3d(dx, dy, yaw); 
        }

        // std::cerr << best_id: " << best_id << "curr_id: " << curr_id << std::endl;
        // std::cerr << "best_pose1: " << best_relative_pose[0] << "best_pose2: " <<  best_relative_pose[1] << "best_pose3: " << best_relative_pose[2] << "deg: " << best_relative_pose[2] * 180.0 / M_PI << std::endl;
    } 

    if (best_id < 0) {
        //  std::cerr << "[LoopClosure] No valid loop candidate found." << std::endl;
        return;
    } 

    
    auto* v1 = dynamic_cast<g2o::VertexSE2*>(slam->graph->vertex(best_id));
    auto* v2 = dynamic_cast<g2o::VertexSE2*>(slam->graph->vertex(curr_id));

    if (!v1) {
        //  std::cerr << "[LoopClosure] Invalid v1 vertices: best_id= " << best_id << std::endl;
        return;
    }

    // std::cerr << "step 6" << std::endl;
    if (!v2) {
        //  std::cerr << "[LoopClosure] Invalid v2 vertices: curr_id= " << curr_id << std::endl;
        return;
    }
       
    auto* edge = add_se2_edge(v1, v2, best_relative_pose, Eigen::Matrix3d::Identity());

    if (!edge) {
        std::cerr << "[LoopClosure] Failed to add loop edge." << std::endl;
        return;
    }
    // std::cerr << "v: " << graph->vertices().size() << "e: " << graph->edges().size());

    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(1.0);
    edge->setRobustKernel(rk);
   
}

}  // namespace graph_slam