#ifndef GRAPH_SLAM_START_2D_HPP
#define GRAPH_SLAM_START_2D_HPP

#include <slam_algorithm.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <message_filters/subscriber.h>
#include <tf2_ros/message_filter.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <fstream> 
#include <string> 
#include <vector> 
#include <iomanip>
#include <chrono>

namespace graph_slam
{

class GraphSlamNode : public rclcpp::Node
{
public:

   struct ScanPoseData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan;
    rclcpp::Time time;
    Eigen::Vector3d pose;
    };
    
    ScanPoseData data;
    std::vector<ScanPoseData> history_;
/**
 * @brief GraphSlamNode를 생성한다 (ROS 2 Foxy).
 *
 * 이 생성자의 역할:
 *  - 노드 이름과 핵심 상태 플래그를 초기화한다.
 *  - std::shared_ptr<rclcpp::Node>가 필요한 API에서 사용할 수 있도록
 *    shared_ptr 형태의 "node_" 핸들을 생성한다.
 *  - TF2 인프라를 설정한다:
 *     - map -> odom 변환을 퍼블리시하기 위한 TransformBroadcaster
 *     - TF 조회를 위한 Buffer와 TransformListener
 *     - TF에서 timeout 기반 transform 요청을 처리하기 위한 Timer 인터페이스
 *  - 다음과 같은 ROS 파라미터를 선언하고 읽어온다:
 *     - 스캔 처리 간격(throttling), 프레임 ID, TF 퍼블리시 주기, 맵 업데이트 주기
 *     - 맵의 범위(xmin/ymin/xmax/ymax)와 해상도(delta)
 *  - Graph SLAM 백엔드(g2o solver 설정)를 생성한다.
 *  - map -> odom 변환을 단위 변환(identity)으로 초기화하고,
 *    아직 초기화가 완료되지 않았음을 표시한다.
 * 
 *  - TF 트리는 다음과 같은 프레임 간 변환을 제공한다:
 *      odom_frame_ <-> base_frame_ <-> laser_frame_
 *      (laser_frame_은 들어오는 스캔 메시지로부터 결정됨)
 *  - 맵은 map_frame_ 기준으로 생성되며,
 *    map_to_odom_ 변환(SLAM 보정)을 통해 odom_frame_과 연결된다.
 *
 * @param options 컴포넌트 컨테이너 또는 rclcpp::spin에서 전달되는 Node 옵션.
 *
 * @note
 * - node_는 `this`를 참조하는 shared_ptr로 생성되며,
 *   no-op deleter(삭제 동작을 수행하지 않는 삭제자)를 사용해
 *   객체가 중복 해제(double deletion)되는 것을 방지한다.
 * 
 * - tf_delay_는 transform_publish_period_와 동일하게 설정되어,
 *   TF 변환의 타임스탬프를 약간 미래 시점으로 찍도록 한다.
 */
    explicit GraphSlamNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    
/**
 * @brief GraphSlamNode의 소멸자.
 *
 * 객체가 파괴되기 전에 백그라운드에서 실행 중이던 TF 퍼블리싱 스레드가
 * 정상적으로 종료되도록 join()을 호출한다.
 * delete로 객체 해제하고 nullptr로 설정한다.
 */
   ~GraphSlamNode(); 

/**
 * @brief 라이브 SLAM 실행에 필요한 ROS2 인터페이스를 초기화한다.
 *
 * 이 메서드는 다음 구성 요소들을 설정한다:
 *
 *  - 퍼블리셔(Publishers):
 *    - "map" (nav_msgs::msg::OccupancyGrid)
 *      → transient_local QoS를 사용하여 늦게 구독한 노드도
 *        마지막으로 발행된 맵을 받을 수 있도록 한다.
 *
 *    - "map_metadata" (nav_msgs::msg::MapMetaData)
 *      → 동일하게 transient_local QoS 사용.
 *
 *  - 서비스(Service):
 *    - "dynamic_map" (nav_msgs::srv::GetMap)
 *      → 현재 생성된 맵을 요청 시 반환한다.
 *
 *  - TF 기반 LaserScan 구독:
 *    - message_filters::Subscriber와 tf2_ros::MessageFilter를 사용하여
 *      필요한 TF 변환이 가능할 때만 스캔이 처리되도록 보장한다.
 *
 *  - 고정된 주기로 TF를 퍼블리시하기 위한 전용 스레드.
 *
 * TF 게이팅(TF gating):
 *  - scan_filter_는 출력 기준 프레임으로 odom_frame_을 사용한다.
 *  - TF를 통해 odom_frame_으로 변환 가능한 스캔만
 *    laserCallback()을 호출하게 된다.
 *
 * @note
 * - message_filters 컴포넌트에서 NodeBase로 node_를 사용하므로,
 *   이 메서드를 호출하기 전에 node_가 반드시 유효해야 한다.
 *
 * - tf2_ros::MessageFilter를 생성하기 전에
 *   tf_buffer_가 반드시 초기화되어 있어야 한다.
 *
 */
    void startLiveSlam();

/**
 * @brief TF를 조회하여 특정 시각에서 odom 프레임 기준 센서의 자세(pose)를 얻는다.
 *
 * 이 함수는 미리 정의된 @c centered_laser_pose
 * (센서 프레임 기준이며, 방향 규약이 적용된 pose)를
 * 주어진 타임스탬프에서 @c odom_frame_으로 변환한다.
 *
 * 처리 단계:
 *  1) 요청된 시간에 맞게 centered_laser_pose의 타임스탬프를 설정한다.
 *  2) tf_buffer_를 사용하여 centered_laser_pose를 odom_frame_으로 변환한다.
 *  3) 변환 결과 쿼터니언에서 yaw 값을 추출한다.
 *  4) (x, y, yaw) 형태의 2D pose를 @p map_pose에 저장한다.
 *
 * @param[out] map_pose 출력용 2차원 pose (x, y, yaw).
 *                      **변수 이름과 달리 이 pose는 odom 프레임 기준이다.**
 * @param[in]  t TF를 평가할 시각 (일반적으로 scan.header.stamp 사용).
 *
 * @return 변환이 성공하고 pose 계산이 완료되면 true.
 * @return TF 조회 또는 변환에 실패하면 false (해당 스캔은 건너뛰어야 함).
 *
 * @note
 * - 반환되는 pose는 odom 프레임 기준이다.
 *
 * - 호출 전에 centered_laser_pose.header.frame_id가 initMapper안에서 laser_frame으로 정의됨.
 */

    bool getOdomPose(Eigen::Vector3d& map_pose, const rclcpp::Time& t);

/**
 * @brief 새로운 LaserScan을 Graph SLAM 파이프라인에 추가한다.
 *
 * 이 함수는 Graph SLAM의 프론트엔드(front-end) 처리를 수행한다:
 *  - 입력된 스캔 데이터의 유효성을 검사한다.
 *  - 과거 스캔 이력을 유지한다.
 *  - 오도메트리 추정값을 기반으로 포즈 그래프에 새로운 SE2(병진과 회전을 표현하는 군) 노드를 생성한다.
 *  - 스캔 매칭을 통해 상대 변환(relative transformation)을 계산한다.
 *  - 연속된 포즈 사이에 edge 제약(constraint)을 추가한다.
 *  - 충분한 스캔 이력이 쌓이면 루프 클로저(loop closure)를 탐지한다.
 *  - 그래프 최적화를 수행하도록 트리거한다.
 *
 * 파이프라인:
 *  LaserScan → PointCloud → Scan Matching → Pose Graph Node → Edge → Optimize
 *
 * @param scan 입력 LaserScan 메시지.
 * @param odom_pose odom 프레임 기준 로봇의 pose (x, y, yaw).
 *
 * @return 스캔이 그래프에 성공적으로 통합되면 true.
 * @return 유효성 검사에 실패하거나 그래프 삽입에 실패하면 false.
 *
 * @note
 * - 스캔 데이터는 TF 조회를 통해 이미 motion compensation이 적용되었다고 가정한다.
 * - 전역 일관성을 개선하기 위해 주기적으로 그래프 최적화를 수행한다.
 * - 일정량 이상의 스캔이 누적되면 루프 클로저를 시도한다.
 */

    bool addScan(const sensor_msgs::msg::LaserScan& scan, Eigen::Vector3d& odom_pose);

/**
 * @brief LaserScan 콜백 함수 (SLAM 파이프라인의 주요 진입 지점).
 *
 * 이 콜백은 tf2_ros::MessageFilter에 의해 호출되며,
 * 들어온 스캔 데이터를 @c odom_frame_으로 변환할 수 있을 때만 실행된다.
 *
 * 파이프라인:
 *  1) CPU 부하를 줄이기 위해 스캔을 간헐적으로 처리(throttling)한다.
 *  2) LaserScan을 PointCloud(current_scan)로 변환한다.
 *  3) 첫 번째 스캔일 경우 mapper를 초기화한다.
 *     (센서 장착 방향 및 centered laser pose 설정).
 *  4) TF를 사용하여 스캔 시점에서 센서의 odom 기준 pose를 계산한다.
 *  5) 스캔을 pose-graph SLAM에 추가한다:
 *     - 노드 추가
 *     - 엣지 추가 (스캔 매칭)
 *     - 루프 클로저 탐지
 *     - 그래프 최적화
 *  6) 이후 소비 노드들이 사용할 수 있도록 map → odom TF 보정(map_to_odom_)을 갱신한다.
 *  7) 제어된 주기에 따라 Occupancy Grid 맵을 업데이트한다.
 *
 * @param scan 입력 LaserScan 데이터.
 *
 * @note
 *
 * - map_to_odom_이 TF 조회(map_frame_ → odom_frame_)를 통해 설정되고 있다.
 *   그러나 일반적인 SLAM 구조에서는 최적화된 pose/그래프와
 *   odom pose를 기반으로 map_to_odom_을 계산한다.
 *
 */

    void laserCallback(const std::shared_ptr<const sensor_msgs::msg::LaserScan> scan);

/**
 * @brief 레이저/센서와 관련된 기하학적 가정을 초기화하고,
 *        중심 기준(centered) 레이저 pose를 캐싱한다.
 *
 * 이 메서드는 SLAM 수행을 위해 노드를 다음과 같이 준비한다:
 *  - 첫 번째 스캔으로부터 레이저 프레임 ID를 기록한다.
 *  - TF를 사용하여 레이저의 장착 방향(평면 장착 여부, 수평인지 아닌지)을 판단한다:
 *    - base 프레임에서 "up" 포인트 (0, 0, 1 + laser_z)를 생성한 뒤
 *      이를 laser 프레임으로 변환한다.
 *    - 변환된 Z 값이 +1 또는 -1에 충분히 가깝지 않다면
 *      레이저가 평면에 장착되지 않은 것으로 간주하고 초기화를 중단한다.
 *
 *  -  입력 데이터의 좌표계가 일관적일수록 안정하다. 센서가 실제로는 조금 돌아가 있을 수도 있다.
 *     매번 센서 기준이 살짝 다르면 SLAM에서 drift가 생길 수 있으므로 scan의 중앙 방향인 "centered" 레이저 pose를 계산한다:
 *    - 스캔의 중앙 각도(center angle)를 사용하여
 *      스캔 각도가 좌우 대칭이 되도록 센서 방향을 정렬한다.
 *    - 정방향 장착과 뒤집힌(upside-down) 장착 모두를 처리한다.
 *
 *  - 스캔 시점에서의 초기 odom pose를 계산한다
 *    (SLAM 초기 상태 설정을 위함).
 *
 * @param[in] scan 초기화에 사용되는 첫 번째 LaserScan.
 *
 * @return 초기화에 성공하고 레이저가 평면에 올바르게 장착되어 있으면 true.
 * @return TF 변환에 실패하거나 센서가 평면 장착이 아니면 false.
 *
 * @note
 * - 다음 프레임 간 TF 변환이 반드시 존재해야 한다:
 *    - laser_frame_ → base_frame_  (base 기준 레이저 위치 계산)
 *    - base_frame_ → laser_frame_ ( "up" 벡터를 통한 방향 검증 )
 *
 * - laser_beam_count_는 scan.ranges.size()로 설정되며,
 *   이후 스캔 데이터의 무결성을 검증하는 데 사용된다.
 *
 * - centered_laser_pose.header.frame_id는 laser_frame_으로 설정되며,
 *   이후 TF 조회는 이 프레임을 기준으로 수행된다.
 *
 * - 계산된 initialPose는 현재 저장되지 않는다.
 *   SLAM 백엔드에서 필요하다면 전달하거나 별도로 저장하는 것을 고려하라.
 */

    bool initMapper(const sensor_msgs::msg::LaserScan& scan);

/**
 * @brief 2D LaserScan 데이터를 평면상의 PCL 포인트 클라우드(XY 평면)로 변환한다.
 *
 * 각 유효한 거리 측정값은 극좌표 → 직교좌표 변환을 통해 다음과 같이 계산된다:
 *   x = r * cos(theta)
 *   y = r * sin(theta)
 *   z = 0
 *
 * @param scan 입력 LaserScan (shared pointer).
 * @return 새롭게 할당된 pcl::PointCloud<pcl::PointXYZ>::Ptr 객체.
 *
 * @note
 *
 * - 생성된 포인트 클라우드는 laser 프레임 좌표계를 기준으로 한다.
 */

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserScanToPointCloud(const sensor_msgs::msg::LaserScan& scan);

/**
 * @brief OccupancyGrid 맵을 PGM 이미지와 YAML 파일로 저장한다.
 *
 * 이 함수는 ROS2의 OccupancyGrid 맵을 Navigation Stack에서
 * 사용할 수 있는 표준 맵 파일 형식으로 변환하여 저장한다.
 *
 * 다음 두 개의 파일이 생성된다.
 *  - `<prefix>.pgm`  : grayscale 형태의 occupancy map 이미지
 *  - `<prefix>.yaml` : 맵의 해상도와 origin 등의 메타데이터
 *
 * OccupancyGrid 값은 다음 규칙에 따라 PGM grayscale 값으로 변환된다.
 *
 * | OccupancyGrid 값 | 의미        | PGM 값 |
 * |------------------|-------------|--------|
 * | -1               | Unknown     | 205 (회색) |
 * | >= 90            | Occupied    | 0 (검정) |
 * | <= 5             | Free space  | 254 (흰색) |
 * | 그 외 값         | 불확실 영역 | 205 (회색) |
 *
 * ROS 맵 파일 좌표계와 맞추기 위해 PGM 파일을 저장할 때
 * y축 방향을 뒤집어서 기록한다.
 *
 * @param map    저장할 OccupancyGrid 맵
 * @param prefix 출력 파일 이름의 prefix (확장자 제외)
 *
 * 사용 예:
 * @code
 * saveMapToPGMAndYAML(map_, "/colcon_ws/2d_graph_map");
 * @endcode
 *
 * 위 코드를 실행하면 다음 파일이 생성된다.
 *  - /colcon_ws/2d_graph_map.pgm
 *  - /colcon_ws/2d_graph_map.yaml
 *
 * @return 파일 저장이 성공하면 true
 * @return 맵 데이터가 비어 있거나 파일 저장에 실패하면 false
 */
bool saveMapToPGMAndYAML(
    const nav_msgs::msg::OccupancyGrid & map,
    const std::string & prefix);

/**
 * @brief 현재 SLAM 맵을 주기적으로 파일로 저장하는 타이머 콜백 함수
 *
 * 이 함수는 ROS2 timer callback으로 사용되며,
 * `got_map_`이 true인지 확인하여 유효한 맵이 생성된 경우
 * `saveMapToPGMAndYAML()` 함수를 이용해 맵을 디스크에 저장한다.
 *
 * 맵은 다음 위치에 저장된다.
 *
 *  /colcon_ws/2d_graph_map.pgm
 *  /colcon_ws/2d_graph_map.yaml
 *
 * 맵 데이터를 안전하게 접근하기 위해 mutex lock을 사용한다.
 *
 * 일반적인 사용 예:
 *
 * @code
 * save_timer_ = this->create_wall_timer(std::chrono::seconds(10), std::bind(&GraphSlamNode::timerSaveMap, this));
 * 
 * @endcode
 *
 * 위와 같이 설정하면 10초마다 자동으로 맵이 저장된다.
 */
void timerSaveMap();                   

/**
 * @brief 전용 루프에서 map → odom 변환을 주기적으로 퍼블리시한다.
 *
 * 이 메서드는 ROS가 실행 중인 동안 무한 루프를 돌며,
 * @p transform_publish_period에 정의된 주기로 publishTransform()을 호출한다.
 *
 * 일반적인 사용 방식:
 *  - startLiveSlam()에서 백그라운드 스레드로 실행된다.
 *  - RViz, Nav2 등 TF를 사용하는 노드들을 위해
 *    지속적으로 최신 TF 트리를 유지한다.
 *
 * @param transform_publish_period TF 퍼블리시 주기 (초 단위).
 *
 * @note
 * - @p transform_publish_period가 0이면
 *   루프는 즉시 종료되며 TF 퍼블리싱이 비활성화된다.
 *
 * - 이 루프는 블로킹(blocking) 방식이므로
 *   별도의 스레드에서 실행되어야 한다.
 *
 * - publishTransform()은 공유 TF 상태(예: map_to_odom_)에 대해
 *   반드시 스레드 안전(thread-safe)하게 구현되어야 한다.
 */

    void publishLoop(double transform_publish_period);

/**
 * @brief 현재 map → odom 변환을 TF로 브로드캐스트한다.
 *
 * map_to_odom_mutex_로 보호되는 @c map_to_odom_을 사용하여
 * 다음과 같은 TransformStamped 메시지를 퍼블리시한다:
 *
 *  - header.frame_id  = map_frame_
 *  - child_frame_id   = odom_frame_
 *  - header.stamp     = now + tf_delay_
 *
 * @note
 * - tf_delay_는 미래 시점의 타임스탬프를 설정하기 위해 사용되며,
 *   TF를 사용하는 노드들이 지연(latency)을 더 잘 허용하고
 *   extrapolation 오류를 줄이는 데 도움을 준다.
 *
 * - map_to_odom_은 일관되게 갱신되어야 하며,
 *   일반적으로 SLAM 최적화 결과를 기반으로 업데이트된다.
 *
 */
    void publishTransform();

/**
 * @brief 최신 최적화 pose와 최근 스캔을 사용하여 Occupancy Grid 맵을 업데이트한다.
 *
 * 이 함수는 다음 작업을 수행한다:
 *  - 첫 호출 시 map_을 지연 초기화(lazy initialization)한다
 *    (해상도, 경계, 원점, 저장 공간).
 *
 *  - SLAM 백엔드로부터 최적화된 pose (x, y, theta)를 조회한다.
 *
 *  - 마지막으로 저장된 스캔(past_scans_.back())을 순회하며 각 히트 포인트에 대해:
 *     - 레이저 프레임 → 맵 프레임으로 좌표 변환
 *     - 맵 프레임 기준 로봇 위치 계산
 *     - 두 좌표를 맵 그리드 인덱스로 변환
 *     - drawLine()을 호출하여 log-odds 공간에서 레이 트레이싱 업데이트 수행
 * 
 *  - ROS OccupancyGrid 형식(0~100)에 맞게 확률을 고려하여 맵에 장애물값, free-cell값을 설정한다.
 *  - OccupancyGrid와 MapMetaData를 퍼블리시한다.
 *
 * @param scan 주로 타임스탬프 및 TF 조회에 사용되는 LaserScan.
 *
 * @thread_safety
 * - 서비스 호출 등과의 동시 접근을 방지하기 위해
 *   map_mutex_ (scoped_lock)로 보호된다.
 *
 * @note
 *
 */
    void drawLineGlobal(int x0_sub, int y0_sub, int x1_sub, int y1_sub);
    void updateMap(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan, const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan,
                              const Eigen::Vector3d& pose);

/**
 * @brief Bresenham 알고리즘과 log-odds를 사용하여
 *        로봇 셀과 히트 셀 사이를 레이 트레이싱 업데이트한다.
 * 
 * 레이 트레이싱: 센서에서 출발한 직선을 따라가면서 공간이 비어있는지, 장애물이 있는지를 판단하는 기법.
 *
 * 시작 셀(로봇)과 끝 셀(레이저 히트)이 주어지면 다음을 수행한다:
 *
 *  - Bresenham 선 그리기 알고리즘으로 광선 경로상의 모든 셀을 순회한다.
 *  - 중간 셀은 자유 공간(l_free)으로 표시한다.
 *  - 끝점 셀은 점유(l_occ)로 표시한다.
 *  - log-odds 값을 [lmin, lmax] 범위로 제한한다.
 *  - log-odds를 확률로 변환하여 OccupancyGrid에 기록한다.
 *
 * @param x0 로봇 셀의 x 인덱스.
 * @param y0 로봇 셀의 y 인덱스.
 * @param x1 레이저를 map좌표로 변환한 셀의 x 인덱스.
 * @param y1 레이저를 map좌표로 변환한 셀의 y 인덱스.
 * @param scan 각도/거리 계산에 사용되는 원본 LaserScan.
 * @param laser_to_map 레이저tf을 맵tf로 변환했을때의 tf laser_to_map.
 * 
 * @note
 */
    void mergeSubmap(nav_msgs::msg::OccupancyGrid& global_map,
                                const nav_msgs::msg::OccupancyGrid& local_submap);
    void drawLine(int x0, int y0, int x1, int y1);

private:
    // Node interfaces are inherited from rclcpp::Node

    // ROS2 publishers/subscribers/services
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_, submap_pub_;
    rclcpp::Publisher<nav_msgs::msg::MapMetaData>::SharedPtr map_metadata_pub_;
    rclcpp::Service<nav_msgs::srv::GetMap>::SharedPtr map_service_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Node::SharedPtr node_;

    // SLAM and map
    std::shared_ptr<graph_slam::GraphSLAM> slam_; 
    rclcpp::TimerBase::SharedPtr save_timer_;

    // TF2
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    tf2::Transform map_to_odom_;

    // tf2 message filter for laser scan
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> scan_filter_sub_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>> scan_filter_;

    // Threading/locking
    boost::thread* transform_thread_;
    boost::mutex map_mutex_;
    boost::mutex map_to_odom_mutex_;

    nav_msgs::msg::OccupancyGrid map_, submap;
    std::vector<float> log_odds_map, log_odds_submap;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> past_scans_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_scan;
    geometry_msgs::msg::PoseStamped centered_laser_pose;
    Eigen::Vector3d first_pose, previous_pose;

    // State
    bool got_map_;
    bool got_first_scan_;
    bool startDetect;
    tf2::Duration map_update_interval_;

    // Frame ids
    std::string base_frame_;
    std::string laser_frame_;
    std::string map_frame_;
    std::string odom_frame_;

    // Params, counters
    int laser_count_;
    int throttle_scans_;
    unsigned int laser_beam_count_;

    double xmin_;
    double ymin_;
    double xmax_;
    double ymax_;
    double delta_;
    double transform_publish_period_;
    double tf_delay_;    
};

} // namespace graph_slam

#endif // GRAPH_SLAM_START_2D_HPP