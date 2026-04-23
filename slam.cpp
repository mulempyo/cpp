#include <slam_algorithm.h>
#include <slam.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2/utils.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <queue>
#include <time.h>
#include <thread>
#include <mutex>
#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

namespace graph_slam {

GraphSlamNode::GraphSlamNode(const rclcpp::NodeOptions &options)
: Node("graph_slam_node", options),
  got_map_(false),
  map_update_interval_(tf2::durationFromSec(0.5))
{
    RCLCPP_INFO(this->get_logger(), "Initializing GraphSlamNode (Foxy version)");
    node_ = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
    // tf2
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(),
    this->get_node_timers_interface()
    );

    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // declare parameters
    throttle_scans_ = this->declare_parameter<int>("throttle_scans", 1);
    base_frame_     = this->declare_parameter<std::string>("base_frame", "base_link");
    map_frame_      = this->declare_parameter<std::string>("map_frame", "map");
    odom_frame_     = this->declare_parameter<std::string>("odom_frame", "odom");
    transform_publish_period_ = this->declare_parameter<double>("transform_publish_period", 0.05);
    double tmp = this->declare_parameter<double>("map_update_interval", 5.0);

    xmin_ = this->declare_parameter<double>("xmin", -100.0);
    ymin_ = this->declare_parameter<double>("ymin", -100.0);
    xmax_ = this->declare_parameter<double>("xmax", 100.0);
    ymax_ = this->declare_parameter<double>("ymax", 100.0);
    delta_ = this->declare_parameter<double>("delta", 0.05);
    tf_delay_ = transform_publish_period_;

    slam_ = std::make_shared<graph_slam::GraphSLAM>("lm_var_csparse");

    map_to_odom_.setIdentity();

    got_first_scan_ = false;
}

GraphSlamNode::~GraphSlamNode() {
    if(transform_thread_) {
        transform_thread_->join();
        delete transform_thread_;
        transform_thread_ = nullptr;
    }
}

void GraphSlamNode::publishLoop(double transform_publish_period) {
    if (transform_publish_period == 0)
        return;
    rclcpp::Rate r(1.0 / transform_publish_period);
    while (rclcpp::ok()) {
        publishTransform();
        r.sleep();
    }
}

void GraphSlamNode::startLiveSlam() {

  map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
  "map",
  rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local()
  );

  map_metadata_pub_ = this->create_publisher<nav_msgs::msg::MapMetaData>(
  "map_metadata",
  rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local()
  );

  submap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
  "submap",
  rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local()
  );

  //LaserScan도착 -> TF 존재 ? callback실행 : 대기 
  scan_filter_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>
            (node_, "scan", rclcpp::SensorDataQoS().get_rmw_qos_profile()); // message_filters 파이프라인에 붙일 수 있는 Subscriber를 만든다.

  scan_filter_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>
            (*scan_filter_sub_, *tf_buffer_, odom_frame_, 10, node_); // 들어온 LaserScan을 odom_frame 으로 TF 변환 가능 여부로 걸러주는 필터.

  scan_filter_->registerCallback(std::bind(&GraphSlamNode::laserCallback, this, std::placeholders::_1)); // MessageFilter가 이 스캔은 odom_frame_으로 변환 가능다고 판단한 순간에 호출할 함수를 등록.

  if (!scan_filter_) {
    RCLCPP_ERROR(this->get_logger(), "scan_filter_ is NULL!");
    return;
  }

  save_timer_ = this->create_wall_timer(std::chrono::seconds(10), std::bind(&GraphSlamNode::timerSaveMap, this));

  // TF2 transform publish thread
   transform_thread_ = new boost::thread(boost::bind(&GraphSlamNode::publishLoop, this, transform_publish_period_));
}

bool GraphSlamNode::getOdomPose(Eigen::Vector3d& map_pose, const rclcpp::Time& t) {

    centered_laser_pose.header.stamp = t;

    geometry_msgs::msg::PoseStamped odom_pose;
    try
    {
        odom_pose = tf_buffer_->transform(
            centered_laser_pose,       
            odom_frame_             
        );
    }
    catch (const tf2::TransformException& e)
    {
        RCLCPP_WARN(this->get_logger(), "Failed to compute odom pose, skipping scan (%s)", e.what());
        return false;
    }

    double yaw = tf2::getYaw(odom_pose.pose.orientation);

    map_pose = Eigen::Vector3d(
        odom_pose.pose.position.x,
        odom_pose.pose.position.y,
        yaw
    );

    return true;
}

bool GraphSlamNode::initMapper(const sensor_msgs::msg::LaserScan &scan)
{
    laser_frame_ = scan.header.frame_id;
    geometry_msgs::msg::PoseStamped ident;
    ident.header.stamp = scan.header.stamp;
    ident.header.frame_id = laser_frame_; 
    tf2::Transform transform;
    transform.setIdentity();
    tf2::toMsg(transform, ident.pose);

    geometry_msgs::msg::PoseStamped laser_pose;
    try {
        laser_pose = tf_buffer_->transform(ident, base_frame_);
    } catch (const tf2::TransformException &e) {
        RCLCPP_WARN(this->get_logger(), "Failed to compute laser pose, aborting initialization (%s)", e.what());
        return false;
    }
    // 센서가 수평하게 장착되었는지 확인하는 검증용 벡터
    geometry_msgs::msg::PointStamped up;
    up.header.stamp = scan.header.stamp;
    up.header.frame_id = base_frame_;
    up.point.x = 0.0;
    up.point.y = 0.0;
    up.point.z = 1.0 + laser_pose.pose.position.z; // 레이저는 바닥보다 위에 있다. 레이저 위치에서 위쪽 방향을 나타내는 벡터를 만들기 위해 1.0m를 더하는 것
    //방향을 알려면 최소 두 점이 필요하다. 예) laser 위치 (0.3m), laser보다 1m 위 (1.3m) -> 두 점을 연결하면 방향 벡터 생성. ↑

    try {
        up = tf_buffer_->transform(up, laser_frame_);
        RCLCPP_DEBUG(this->get_logger(), "Z-Axis in sensor frame: %.3f", up.point.z);
    } catch (const tf2::TransformException &e) {
        RCLCPP_WARN(this->get_logger(), "Unable to determine orientation of laser: %s", e.what());
        return false;
    }

    if (fabs(fabs(up.point.z) - 1) > 0.001) {
        RCLCPP_WARN(this->get_logger(), "Laser has to be mounted planar! Z-coordinate has to be 1 or -1, but gave: %.5f", up.point.z);
        return false;
    }

    laser_beam_count_ = static_cast<unsigned int>(scan.ranges.size());

    double angle_center = (scan.angle_min + scan.angle_max) / 2.0;

    tf2::Quaternion q;

    if (up.point.z > 0) {
        q.setRPY(0, 0, angle_center);
        RCLCPP_INFO(this->get_logger(), "Laser is mounted upwards.");
    } else {
        q.setRPY(M_PI, 0, -angle_center);
        RCLCPP_INFO(this->get_logger(), "Laser is mounted upside down.");
    }

    centered_laser_pose.header.stamp = this->now();
    centered_laser_pose.header.frame_id = laser_frame_; 

    centered_laser_pose.pose.position.x = 0;
    centered_laser_pose.pose.position.y = 0;
    centered_laser_pose.pose.position.z = 0;

    centered_laser_pose.pose.orientation.w = q.getW();
    centered_laser_pose.pose.orientation.x = q.getX();
    centered_laser_pose.pose.orientation.y = q.getY();
    centered_laser_pose.pose.orientation.z = q.getZ();

    Eigen::Vector3d initialPose;
    if (!getOdomPose(initialPose, scan.header.stamp)) {
        RCLCPP_WARN(this->get_logger(), "Unable to determine initial pose of laser! Starting point will be set to zero.");
        initialPose = Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    return true;
}

bool GraphSlamNode::addScan(const sensor_msgs::msg::LaserScan& scan, Eigen::Vector3d& odom_pose)
{
    if (scan.ranges.size() != laser_beam_count_) {
        return false;
    }

    if(slam_->num_vertices() == 0){
        g2o::VertexSE2* first_node = slam_->add_se2_node(odom_pose);
        first_pose = odom_pose;
        if(!first_node){
            RCLCPP_ERROR(node_->get_logger(), "Failed to add fisrt node to graph!");
            return false;
      }

      first_node->setFixed(true);
      past_scans_.push_back(current_scan);
      return true;
    }

    if (slam_->num_vertices() >= 1 && !past_scans_.empty()) {
        int prev_id = slam_->num_vertices() - 1;

        if (prev_id < 0) {
            RCLCPP_ERROR(this->get_logger(), "Invalid previous node ID: %d, skipping edge creation.", prev_id);
            return false;
        }

        g2o::VertexSE2* prev_node = dynamic_cast<g2o::VertexSE2*>(slam_->getGraph()->vertex(prev_id));

        if (prev_node == nullptr) {
            return false;
        }

        if (!prev_node) {
            RCLCPP_ERROR(this->get_logger(), "Previous node (ID: %d) is NULL, skipping edge creation.", prev_id);
            return false;
        }

        Eigen::Vector3d relative_pose = slam_->compute_scan_matching(current_scan, past_scans_.back(), odom_pose, previous_pose);

        //RCLCPP_WARN(node_->get_logger(), "rdx:%f, rdy:%f, rdth:%f", relative_pose[0], relative_pose[1], relative_pose[2]);

        g2o::VertexSE2* new_node = slam_->add_se2_node(odom_pose);

        if (new_node == nullptr) {
          RCLCPP_WARN(node_->get_logger(), "new_node nullptr");
          return false;
        }

      if (!new_node) {
          RCLCPP_ERROR(node_->get_logger(), "Failed to add new node to graph!");
          return false;
      }

        slam_->add_se2_edge(prev_node, new_node, relative_pose, Eigen::Matrix3d::Identity());
    }

    past_scans_.push_back(current_scan);

    double dx = odom_pose.x() - first_pose.x();
    double dy = odom_pose.y() - first_pose.y();
    double dist = std::hypot(dx, dy);

    if(dist >= 1.0 && !startDetect){ 
        RCLCPP_WARN(node_->get_logger(), "ready to start detect loop");
        startDetect = true;
    }

    int node_size = slam_->graph->vertices().size();

    if (node_size >= 100 && startDetect) {
        slam_->detect_loop_closure(slam_, past_scans_, current_scan, odom_pose, previous_pose);
    }

    previous_pose = odom_pose;

    static rclcpp::Time last_optimize_update = this->now();
    if ((this->now() - last_optimize_update).seconds() > 0) {
        slam_->optimize(10);
        last_optimize_update = rclcpp::Time(scan.header.stamp);
    }

    return true;
}


void GraphSlamNode::laserCallback(const std::shared_ptr<const sensor_msgs::msg::LaserScan> scan) {
    laser_count_++;
    if ((laser_count_ % throttle_scans_) != 0)
        return;

    current_scan = laserScanToPointCloud(scan);

    if (current_scan->empty()) {
        return;
    }

    if (!got_first_scan_) {
        if (!initMapper(*scan)) {
            return;
        }
        got_first_scan_ = true;
    }

    Eigen::Vector3d odom_pose;

    if(!getOdomPose(odom_pose, scan->header.stamp)){
        return;
    }

    if(addScan(*scan, odom_pose)){
        Eigen::Vector3d optimized_pose = slam_->getOptimizedPose();
        if (std::isnan(optimized_pose[0]) || std::isnan(optimized_pose[1]) || std::isnan(optimized_pose[2])) {
            RCLCPP_ERROR(this->get_logger(), "Optimized pose contains NaN values! Skipping TF update.");
            return;
        }
       
        //RCLCPP_WARN(node_->get_logger(), "odx:%f, ody:%f, odth:%f", odom_pose[0], odom_pose[1], odom_pose[2]);
        //RCLCPP_WARN(node_->get_logger(), "opx:%f, opy:%f, opth:%f", optimized_pose[0], optimized_pose[1], optimized_pose[2]);

        tf2::Transform map_to_laser_tf, odom_to_laser_tf;
        geometry_msgs::msg::TransformStamped map_to_laser_msg, odom_to_laser_msg;

        tf2::Quaternion q_map, q_odom;
        q_map.setRPY(0.0, 0.0, optimized_pose[2]);
        q_odom.setRPY(0.0, 0.0, odom_pose[2]);

        map_to_laser_tf = tf2::Transform(q_map, tf2::Vector3(optimized_pose[0], optimized_pose[1], 0.0));

        odom_to_laser_tf = tf2::Transform(q_odom, tf2::Vector3(odom_pose[0], odom_pose[1], 0.0));

       
        // map 좌표계에서 본 laser의 pose = map_to_laser_tf
        map_to_odom_mutex_.lock();
        map_to_odom_ = map_to_laser_tf * odom_to_laser_tf.inverse();
        /*tf2::Transform test = map_to_odom_ * odom_to_laser_tf;
        RCLCPP_WARN(node_->get_logger(), "map_to_laser_x:%f, map_to_laser_y:%f, map_to_laser_rot:%f", test.getOrigin().x(), test.getOrigin().y(), tf2::getYaw(test.getRotation()));*/
        map_to_odom_mutex_.unlock();

        static rclcpp::Time last_map_update = this->now();
        if (!got_map_ || (this->now() - last_map_update).seconds() > 0) {
            updateMap(scan, current_scan, optimized_pose);
            last_map_update = rclcpp::Time(scan->header.stamp);
        }
    } else {
        RCLCPP_DEBUG(this->get_logger(), "cannot process scan");
    }
}


pcl::PointCloud<pcl::PointXYZ>::Ptr GraphSlamNode::laserScanToPointCloud(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    double angle = scan->angle_min;
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        if (std::isfinite(scan->ranges[i])) {
            pcl::PointXYZ point;
            point.x = scan->ranges[i] * cos(angle);
            point.y = scan->ranges[i] * sin(angle);
            point.z = 0.0;  

            cloud->push_back(point);
        }
        angle += scan->angle_increment;
    }
    return cloud;
}

void GraphSlamNode::updateMap(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan, const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan,
                              const Eigen::Vector3d& pose) {
  boost::mutex::scoped_lock  map_lock(map_mutex_);

  if (!got_map_) {
      map_.info.resolution = delta_;
      map_.header.frame_id = map_frame_;
      map_.info.width = static_cast<unsigned int>((xmax_ - xmin_) / delta_);
      map_.info.height = static_cast<unsigned int>((ymax_ - ymin_) / delta_);
      map_.info.origin.position.x = xmin_;
      map_.info.origin.position.y = ymin_;
      map_.info.origin.position.z = 0.0;
      map_.info.origin.orientation.w = 1.0;
      map_.data.resize(map_.info.width * map_.info.height);
      map_.data.assign(map_.info.width * map_.info.height, -1); // 전체를 -1로 다시 채운다
      log_odds_map.resize(map_.info.width * map_.info.height); // log_odds_map -> 셀이 점유되었을 가능성 확률을 로그 형태로 저장하는 맵
      log_odds_map.assign(map_.info.width * map_.info.height, 0); // 전체를 0으로 다시 채운다
  }

    pcl::PointCloud<pcl::PointXYZ>::Ptr last_scan = past_scans_.back();
    Eigen::Vector3d posee = slam_->getOptimizedPose();

    double robot_x = posee[0];
    double robot_y = posee[1];
    double robot_theta = posee[2];

  //RCLCPP_INFO(this->get_logger(), "Updating map based on robot pose: x=%f, y=%f, theta=%f", robot_x, robot_y, robot_theta);

  if (!past_scans_.empty()) {
      if (submap.data.empty()) {
        submap.info.resolution = delta_;
        submap.header.frame_id = map_frame_;

        double global_size_x = xmax_ - xmin_;
        double global_size_y = ymax_ - ymin_;
        const double submap_size_x = 5.0;
        const double submap_size_y = 5.0;

        submap.info.width  = static_cast<unsigned int>(submap_size_x / delta_);
        submap.info.height = static_cast<unsigned int>(submap_size_y / delta_);

        submap.info.origin.position.x = robot_x - 0.5 * submap.info.width * delta_;
        submap.info.origin.position.y = robot_y - 0.5 * submap.info.height * delta_;
        submap.info.origin.position.z = 0.0;
        submap.info.origin.orientation.x = 0.0;
        submap.info.origin.orientation.y = 0.0;
        submap.info.origin.orientation.z = 0.0;
        submap.info.origin.orientation.w = 1.0;

        submap.data.assign(submap.info.width * submap.info.height, -1);
        log_odds_submap.assign(submap.info.width * submap.info.height, 0.0f);
      }
     
      double c = std::cos(robot_theta), s = std::sin(robot_theta); 

      for (const auto& point : last_scan->points) {
          // map_to_laser == optimized_pose
          double laserX = robot_x; 
          double laserY = robot_y; 

          int laser_gx = static_cast<int>(std::floor((laserX - map_.info.origin.position.x) / delta_));
          int laser_gy = static_cast<int>(std::floor((laserY - map_.info.origin.position.y) / delta_));

          double worldX = laserX + c * point.x - s * point.y; 
          double worldY = laserY + s * point.x + c * point.y;

          int gx = static_cast<int>(std::floor((worldX - map_.info.origin.position.x) / delta_));
          int gy = static_cast<int>(std::floor((worldY - map_.info.origin.position.y) / delta_));

          //submap.info.origin.position.x = robot_x - 0.5 * submap.info.width * delta_;
          //submap.info.origin.position.y = robot_y - 0.5 * submap.info.height * delta_;

          drawLineGlobal(laser_gx, laser_gy, gx, gy);   
      }

     // submap 영역에 해당하는 global cell만 map_.data 갱신
    int gx0 = std::max(0, (int)((submap.info.origin.position.x - map_.info.origin.position.x) / delta_));
    int gy0 = std::max(0, (int)((submap.info.origin.position.y - map_.info.origin.position.y) / delta_));
    int gx1 = std::min((int)map_.info.width,
                       gx0 + (int)submap.info.width);
    int gy1 = std::min((int)map_.info.height,
                       gy0 + (int)submap.info.height);

    for (int y = 0; y < map_.info.height; ++y) {
        for (int x = 0; x < map_.info.width; ++x) {
            int idx = MAP_IDX(map_.info.width, x, y);

            if (std::abs(log_odds_map[idx]) < 1e-6f) {
                map_.data[idx] = -1;
                continue;
            }

            float odds = std::exp(log_odds_map[idx]);
            float prob = odds / (1.0f + odds);
            int new_value = static_cast<int>(prob * 100.0f);

            if (!got_map_) {
                map_.data[idx] = new_value;
            } else {
                if (map_.data[idx] > 90) {
                    if (prob < 0.03f) {
                        map_.data[idx] = new_value;
                    }
                } else {
                    map_.data[idx] = new_value;
                }
            }
        }
    }
  }

  //map_.data.assign(map_.info.width * map_.info.height, -1);
  //mergeSubmap(map_, submap);

  //RCLCPP_WARN(this->get_logger(),"publishmap");
  got_map_ = true;
  map_.header.stamp = this->now();
  map_.header.frame_id = map_frame_;
  map_pub_->publish(map_);
  map_metadata_pub_->publish(map_.info);

  submap.header.stamp = this->now();
  submap.header.frame_id = map_frame_;
  submap_pub_->publish(submap);
}

void GraphSlamNode::drawLineGlobal(int x0, int y0, int x1, int y1)
{
    const float lmin = -4.0f, lmax = 4.0f;
    const float l_occ = +0.5f;
    const float l_free = -0.5f;

    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    int cx = x0, cy = y0;

    while (true) {
        bool at_endpoint = (cx == x1 && cy == y1);

        if (cx >= 0 && cx < (int)map_.info.width &&
            cy >= 0 && cy < (int)map_.info.height &&
            !(cx == x0 && cy == y0))
        {
            int idx = MAP_IDX(map_.info.width, cx, cy);

            if (!at_endpoint) {
                if (log_odds_map[idx] > 2.0f) {
                    log_odds_map[idx] = std::max(lmin, log_odds_map[idx] + 0.2f * l_free);
                } else {
                    log_odds_map[idx] = std::max(lmin, log_odds_map[idx] + l_free);
                }
            } else {
                log_odds_map[idx] = std::min(lmax, log_odds_map[idx] + l_occ);
                break;
            }
        } else if (at_endpoint) {
            break;
        }

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; cx += sx; }
        if (e2 <  dx) { err += dx; cy += sy; }
    }
}

/* x0:robot x, y0: roboy y, x1: laserpoint x, y1: laserpoint y */
//drawLine: it use Bresenham algorithm

void GraphSlamNode::drawLine(int x0, int y0, int x1, int y1) {
  const double lmin = -4.0, lmax = 4.0; 
  const double l_occ = +0.7;   
  const double l_free = -0.2;  

  int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;

  int cx = x0, cy = y0;
  size_t i = 0;

  while (true) { 
    bool at_endpoint = (cx == x1 && cy == y1); 

    if (!at_endpoint) { 
        if (cx >= 0 && cx < (int)submap.info.width && cy >= 0 && cy < (int)submap.info.height && cx != x0 && cy != y0) { 
            int idx = MAP_IDX(submap.info.width, cx, cy);  
            if (log_odds_submap[idx] > 2.0f){ // 강한 장애물은 천천히만 감소
              log_odds_submap[idx] = std::max(lmin, log_odds_submap[idx] + 0.2f * l_free);
            } else {
              // 일반 셀은 정상적으로 free 반영
              log_odds_submap[idx] = std::max(lmin, log_odds_submap[idx] + l_free);
            }

        }  
    } else { 
        if (cx >= 0 && cx < (int)submap.info.width && cy >= 0 && cy < (int)submap.info.height && cx != x0 && cy != y0) { 
            int idx = MAP_IDX(submap.info.width, cx, cy); 
            log_odds_submap[idx] = std::min(lmax, log_odds_submap[idx] + l_occ); 
        } 
        break; 
    } 
    
    int e2 = 2 * err; 
    if (e2 > -dy) { err -= dy; cx += sx; } 
    if (e2 < dx) { err += dx; cy += sy; } 
  }
  
}

bool GraphSlamNode::saveMapToPGMAndYAML(
    const nav_msgs::msg::OccupancyGrid & map,
    const std::string & prefix)
{
    if (map.info.width == 0 || map.info.height == 0 || map.data.empty()) {
        return false;
    }

    const std::string pgm_file  = prefix + ".pgm";
    const std::string yaml_file = prefix + ".yaml";

    // 1) PGM 저장
    std::ofstream pgm(pgm_file, std::ios::out | std::ios::binary);
    if (!pgm.is_open()) {
        return false;
    }

    pgm << "P5\n";
    pgm << "# CREATOR: GraphSlamNode\n";
    pgm << map.info.width << " " << map.info.height << "\n";
    pgm << "255\n";

    for (int y = static_cast<int>(map.info.height) - 1; y >= 0; --y) {
    for (unsigned int x = 0; x < map.info.width; ++x) {
        const int8_t occ = map.data[y * map.info.width + x];
        uint8_t value = 205;  // 기본 unknown

        if (occ == -1) {
            value = 205;      // unknown -> gray
        } else if (occ >= 90) {
            value = 0;        // occupied -> black
        } else if (occ <= 5) {
            value = 254;      // free -> white
        } else {
            value = 205;      // 애매한 확률은 unknown처럼 회색 처리
        }

        pgm.write(reinterpret_cast<const char*>(&value), 1);
    }
}
    pgm.close();

    // 2) YAML 저장
    std::ofstream yaml(yaml_file);
    if (!yaml.is_open()) {
        return false;
    }

    yaml << "image: " << pgm_file.substr(pgm_file.find_last_of('/') + 1) << "\n";
    yaml << "mode: trinary\n";
    yaml << "resolution: " << std::fixed << std::setprecision(6)
         << map.info.resolution << "\n";
    yaml << "origin: ["
         << map.info.origin.position.x << ", "
         << map.info.origin.position.y << ", 0.0]\n";
    yaml << "negate: 0\n";
    yaml << "occupied_thresh: 0.65\n";
    yaml << "free_thresh: 0.25\n";
    yaml.close();

    return true;
}

void GraphSlamNode::timerSaveMap()
{
    boost::mutex::scoped_lock  map_lock(map_mutex_);

    if (!got_map_) {
        return;
    }
    if (got_map_ && map_.info.width && map_.info.height) {
        RCLCPP_WARN(node_->get_logger(), "2d_graph_map publish");
        saveMapToPGMAndYAML(map_, "/colcon_ws/2d_graph_map");
    }
}

void GraphSlamNode::publishTransform() {
    map_to_odom_mutex_.lock();
    rclcpp::Time tf_expiration = this->now() + rclcpp::Duration(static_cast<int32_t>(static_cast<rcl_duration_value_t>(tf_delay_)), 0);
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = tf_expiration; //tf_expiration을 써야 회전을 할때 맵과 라이다값이 맞지 않는 문제를 해결할 수 있다.
    t.header.frame_id = map_frame_;
    t.child_frame_id = odom_frame_;
    try {
        t.transform = tf2::toMsg(map_to_odom_);
        tf_broadcaster_->sendTransform(t);
    }catch (tf2::LookupException& e){
        RCLCPP_INFO(this->get_logger(), e.what());
    }
    map_to_odom_mutex_.unlock();
    
}

}