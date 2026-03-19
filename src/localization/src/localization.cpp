#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/clock.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/utils.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <thread>



using namespace std::chrono_literals;
constexpr auto TIMEOUT = 1000ms; // pain

// who doesn't love global vars? :DD (it's for debugging only, relax)
constexpr bool DO_ICP = true; // false means we do something else that didn't work at first (skill issue)
constexpr bool deskewing = false;
long int frame = 0;
long int marker_id = 0;

pcl::PointCloud<pcl::PointXYZ> scan_to_pc(const sensor_msgs::msg::LaserScan::SharedPtr scan, const tf2::Transform & T_start, const tf2::Transform & T_end)
{
	/* 
	This function deskews and converts the laser scan readings into a 2D point cloud in the lidar frame.
	*/
	const size_t num_ranges = scan->ranges.size();
	pcl::PointCloud<pcl::PointXYZ> pointcloud;
	pointcloud.reserve(num_ranges); // preallocate

	tf2::Transform T_i;
	tf2::Transform T_start_i;
	tf2::Vector3 p_i;
	tf2::Quaternion q_i;
	float r;
	tf2::Vector3 p_out;

	for (size_t i = 0; i < num_ranges; ++i) 
	{
		float range = scan->ranges[i];
		if (range >= scan->range_min && range <= scan->range_max) 
		{
			// Convert (range,angle) to (x,y,z)
			float angle = scan->angle_min + i * scan->angle_increment;
			float x = range * cos(angle);
			float y = range * sin(angle);
			float z = 0.0;
			
			if (deskewing) 
			{
				// Interpolate lidar point TF
				r = std::clamp((i * scan->time_increment) / ((num_ranges - 1)*scan->time_increment), 0.0f, 1.0f);
				p_i = tf2::lerp(T_start.getOrigin(), T_end.getOrigin(), r);
				q_i = tf2::slerp(T_start.getRotation(), T_end.getRotation(), r);
				T_i.setOrigin(p_i);
				T_i.setRotation(q_i);

				// TF point back to start TF
				T_start_i = T_start.inverseTimes(T_i);
				p_out = T_start_i * tf2::Vector3(x,y,z);
				pointcloud.emplace_back(p_out.getX(), p_out.getY(), p_out.getZ());
			}
			else
			{
				pointcloud.emplace_back(x,y,z);
			}

			
		}
	}
	return pointcloud;
}

pcl::PointCloud<pcl::PointXYZ> filter_cloud(const pcl::PointCloud<pcl::PointXYZ> & input)
{
	/*
	This function filters out outliers in a point cloud
	*/
    pcl::PointCloud<pcl::PointXYZ> filtered;

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(input.makeShared());
    ror.setRadiusSearch(0.20);
    ror.setMinNeighborsInRadius(2);
    ror.filter(filtered);

    return filtered;
}

void visualize_shit(const pcl::PointCloud<pcl::PointXYZ>::Ptr target_pc, const pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc, const pcl::PointCloud<pcl::PointXYZ>::Ptr result_pc) 
{
	// Visualization (modified from https://github.com/PointCloudLibrary/pcl/blob/master/doc/tutorials/content/sources/interactive_icp/interactive_icp.cpp)
	pcl::visualization::PCLVisualizer viewer("ICP demo");
	// Create two vertically separated viewports
	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	// The color we will be using
	float bckgr_gray_level = 0.0;  // Black
	float txt_gray_lvl = 1.0 - bckgr_gray_level;

	// Original point cloud is white
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_pc_color_h(target_pc, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl);
	viewer.addPointCloud(target_pc, target_pc_color_h, "target_pc_v1", v1);
	viewer.addPointCloud(target_pc, target_pc_color_h, "target_pc_v2", v2);

	// Transformed point cloud is green
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> current_pc_color_h(current_pc, 20, 180, 20);
	viewer.addPointCloud(current_pc, current_pc_color_h, "current_pc_v1", v1);

	// ICP aligned point cloud is red
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> result_pc_color_h(result_pc, 180, 20, 20);
	viewer.addPointCloud(result_pc, result_pc_color_h, "result_pc_v2", v2);

	// Adding text descriptions in each viewport
	viewer.addText("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
	viewer.addText("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

	std::stringstream ss;
  	ss << ++frame;
  	std::string frame_cnt = "Frame " + ss.str();
	viewer.addText(frame_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "frame_cnt", v2);

	// Set background color
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

	// Set camera position and orientation
	viewer.setCameraPosition(0, 0, 30, 0, -1.57, 0, 0);
	viewer.setSize(1280, 720);  // Visualiser window size

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	viewer.close();
}

void run_ndt_2d(
	pcl::NormalDistributionsTransform2D<pcl::PointXYZ, pcl::PointXYZ> & ndt,
    const pcl::PointCloud<pcl::PointXYZ> & current_pc,
    const pcl::PointCloud<pcl::PointXYZ> & prev_pc,
    const Eigen::Matrix4f & guess,
	pcl::PointCloud<pcl::PointXYZ> & result_pc) 
{
	// This badboy does NOT work as it is right now, TFs go haywire
    ndt.setInputSource(current_pc.makeShared());
    ndt.setInputTarget(prev_pc.makeShared());
    ndt.setMaximumIterations(50);
    ndt.setTransformationEpsilon(1e-8);

    ndt.align(result_pc, guess);
    return;
}

void run_icp(
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> & icp,
    const pcl::PointCloud<pcl::PointXYZ> & current_pc,
    const pcl::PointCloud<pcl::PointXYZ> & prev_pc,
    const Eigen::Matrix4f & guess,
	pcl::PointCloud<pcl::PointXYZ> & result_pc) 
{
    icp.setInputSource(current_pc.makeShared());
    icp.setInputTarget(prev_pc.makeShared());
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-12);
	icp.setEuclideanFitnessEpsilon(1e-12);
	icp.setMaxCorrespondenceDistance(0.1);

    icp.align(result_pc, guess);
	return;
}

static tf2::Transform tf2_from_msg(const geometry_msgs::msg::TransformStamped & transform_stamped)
{
	tf2::Transform tf;
	tf.setOrigin(tf2::Vector3(
		transform_stamped.transform.translation.x,
		transform_stamped.transform.translation.y,
		transform_stamped.transform.translation.z
	));
	tf.setRotation(tf2::Quaternion(
		transform_stamped.transform.rotation.x,
		transform_stamped.transform.rotation.y,
		transform_stamped.transform.rotation.z,
		transform_stamped.transform.rotation.w
	));
	return tf;
}

static Eigen::Matrix4f tf2_to_eig(const tf2::Transform & transform) 
{
	Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
	const tf2::Matrix3x3& R = transform.getBasis();
	const tf2::Vector3& t = transform.getOrigin();

	matrix(0,0) = R[0][0]; matrix(0,1) = R[0][1]; matrix(0,2) = R[0][2];
	matrix(1,0) = R[1][0]; matrix(1,1) = R[1][1]; matrix(1,2) = R[1][2];
	matrix(2,0) = R[2][0]; matrix(2,1) = R[2][1]; matrix(2,2) = R[2][2];
	matrix(0,3) = static_cast<float>(t.x());
	matrix(1,3) = static_cast<float>(t.y());
	matrix(2,3) = static_cast<float>(t.z());

	return matrix;
}

visualization_msgs::msg::Marker pc_to_marker(const pcl::PointCloud<pcl::PointXYZ> & cloud, rclcpp::Time stamp) 
{
	visualization_msgs::msg::Marker marker;
	marker.header.frame_id = "lidar_link";
	marker.header.stamp = stamp;
	marker.ns = "deskewed_scan";
	marker.lifetime = rclcpp::Duration::from_seconds(1000);
	marker.id = stamp.nanoseconds();
	marker.type = visualization_msgs::msg::Marker::POINTS;
	marker.action = visualization_msgs::msg::Marker::ADD;
	marker.scale.x = 0.01;
	marker.scale.y = marker.scale.x;
	marker.color.a = 0.25;
	marker.color.r = 0.0;
	marker.color.g = 1.0;
	marker.color.b = 1.0;

	for (size_t i = 0; i < cloud.size(); ++i) 
	{
		geometry_msgs::msg::Point p;
		p.x = cloud[i].x;
		p.y = cloud[i].y;
		p.z = cloud[i].z;

		marker.points.push_back(p);

	}

	return marker;
}

visualization_msgs::msg::Marker marker_from_cloud(
	const pcl::PointCloud<pcl::PointXYZ> & current_cloud, 
	const pcl::PointCloud<pcl::PointXYZ> & result_cloud, 
	sensor_msgs::msg::LaserScan::SharedPtr msg, 
	const float r, 
	const float g, 
	const float b,
	const size_t id)
{
	/* 
	Creates a marker in the base frame for visualization purposes.
	*/
	visualization_msgs::msg::Marker marker;
	marker.header.frame_id = "lidar_link";
	marker.header.stamp = msg->header.stamp;
	marker.ns = "icp_debug";
	marker.id = id;
	marker.type = visualization_msgs::msg::Marker::LINE_LIST;
	marker.action = visualization_msgs::msg::Marker::ADD;
	marker.scale.x = 0.01;
	marker.color.a = 1.0;
	marker.color.r = r;
	marker.color.g = g;
	marker.color.b = b;

	for (size_t i = 0; i < std::min<size_t>(current_cloud.size(), result_cloud.size()); ++i) 
	{
		geometry_msgs::msg::Point p_raw, p_aligned;
		p_raw.x = current_cloud[i].x;
		p_raw.y = current_cloud[i].y;
		p_raw.z = current_cloud[i].z;

		p_aligned.x = result_cloud[i].x;
		p_aligned.y = result_cloud[i].y;
		p_aligned.z = result_cloud[i].z;

		marker.points.push_back(p_raw);
		marker.points.push_back(p_aligned);
	}

	return marker;
}

class Localization : public rclcpp::Node
{
	public:
		Localization() : Node("localization")
		{
			
			auto imu_callback = [this](const sensor_msgs::msg::Imu::SharedPtr msg) -> void
			{
				ang_vel_ = msg->angular_velocity.z;
			};

			auto scan_callback = [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) -> void 
			{	

				const auto start_time = this->get_clock()->now().nanoseconds();

				// 1. Get odom pose at current scan timestamp
				tf2::Transform T_odom_curr_lidar, T_odom_old_lidar;
				try
				{
					T_odom_curr_lidar = tf2_from_msg(
						tf_buffer_->lookupTransform("odom", "lidar_link", msg->header.stamp+rclcpp::Duration::from_seconds(deskewing ? msg->scan_time : 0.0), TIMEOUT)
					);
					if (deskewing)
					{
						T_odom_old_lidar = tf2_from_msg(
							tf_buffer_->lookupTransform("odom", "lidar_link", msg->header.stamp, TIMEOUT)
						);
					}
					
				}
				catch (tf2::TransformException & ex)
				{
					RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					if (!map_to_odom_msg_.header.frame_id.empty()) 
					{
						map_to_odom_msg_.header.stamp = msg->header.stamp;
						tf_broadcaster_->sendTransform(map_to_odom_msg_);
					}
					return;
				}

				/*
				if (std::abs(ang_vel_) > 0.3) {
					RCLCPP_INFO(this->get_logger(), "Turning too fast, skipping");
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					if (!map_to_odom_msg_.header.frame_id.empty()) 
					{
						map_to_odom_msg_.header.stamp = msg->header.stamp;
						tf_broadcaster_->sendTransform(map_to_odom_msg_);
					}
					return;
				}
				*/

				// 2. Convert scan to 2D points in lidar frame 
				pcl::PointCloud<pcl::PointXYZ> current_pc = scan_to_pc(msg, T_odom_old_lidar, T_odom_curr_lidar);
				pcl::PointCloud<pcl::PointXYZ> filtered_current_pc = filter_cloud(current_pc);
				rclcpp::Time current_stamp = msg->header.stamp;

				if (filtered_current_pc.empty()) {
					RCLCPP_WARN(this->get_logger(), "Scan produced no valid points; skipping.");
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					if (!map_to_odom_msg_.header.frame_id.empty()) 
					{
						map_to_odom_msg_.header.stamp = msg->header.stamp;
						tf_broadcaster_->sendTransform(map_to_odom_msg_);
					}
					return;
				}

				if (!prev_pc_exists_) 
				{
					// Store first scan
					T_odom_prev_lidar_ = T_odom_curr_lidar;
					filtered_prev_pc_ = filtered_current_pc;
					filtered_first_pc_ = filtered_current_pc;
					T_odom_first_lidar_ = T_odom_prev_lidar_;
					T_map_first_lidar_ = T_map_odom_ * T_odom_first_lidar_;
					prev_pc_exists_ = true;
					RCLCPP_INFO(this->get_logger(), "No previous scan available, cannot run ICP");
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					if (!map_to_odom_msg_.header.frame_id.empty()) 
					{
						map_to_odom_msg_.header.stamp = msg->header.stamp;
						tf_broadcaster_->sendTransform(map_to_odom_msg_);
					}
					return;
				}

				// Publish "corrected" scan
				//pc_publisher_->publish(pc_to_marker(filtered_current_pc, msg->header.stamp));

				// 3. Relative motion prediction from odom
				// source=current, target=prev => guess must be prev <- curr
				
				float rot_diff = static_cast<float>(tf2::getYaw(T_odom_curr_lidar.getRotation()) - tf2::getYaw(T_odom_first_lidar_.getRotation()));
				rot_diff = std::abs(std::atan2(sin(rot_diff), cos(rot_diff))); // wrap it to [0,pi]!
				const float dist_to_start = static_cast<float>(T_odom_curr_lidar.getOrigin().distance(T_odom_first_lidar_.getOrigin()));
				const bool near_start_pose = true; //dist_to_start < 0.3 && rot_diff < 0.3;
				const tf2::Transform tf2_guess = (near_start_pose ? T_odom_first_lidar_ : T_odom_prev_lidar_).inverseTimes(T_odom_curr_lidar);
				const Eigen::Matrix4f guess = tf2_to_eig(tf2_guess);
				Eigen::Matrix4f T;
				double fitness;
				pcl::PointCloud<pcl::PointXYZ> result;
				pcl::PointCloud<pcl::PointXYZ> filtered_target_pc = near_start_pose ? filtered_first_pc_ : filtered_prev_pc_;
				if (DO_ICP) 
				{
					// ICP
					pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
					run_icp(icp, filtered_current_pc, filtered_target_pc, guess, result);
					if (!icp.hasConverged()) 
					{
						RCLCPP_WARN(this->get_logger(), "ICP did not converge");
						// Broadcast old map -> odom TF every time (unless it's time to update it)
						if (!map_to_odom_msg_.header.frame_id.empty()) 
						{
							map_to_odom_msg_.header.stamp = msg->header.stamp;
							tf_broadcaster_->sendTransform(map_to_odom_msg_);
						}
						return;
					}
					T = icp.getFinalTransformation();
					fitness = icp.getFitnessScore();
				}
				else
				{
					// Use NDT2D instead of pure ICP
					pcl::NormalDistributionsTransform2D<pcl::PointXYZ, pcl::PointXYZ> ndt;
					run_ndt_2d(ndt, filtered_current_pc, filtered_target_pc, guess, result);
					if (!ndt.hasConverged()) 
					{
						RCLCPP_WARN(this->get_logger(), "NDT2D did not converge");
						// Broadcast old map -> odom TF every time (unless it's time to update it)
						if (!map_to_odom_msg_.header.frame_id.empty()) 
						{
							map_to_odom_msg_.header.stamp = msg->header.stamp;
							tf_broadcaster_->sendTransform(map_to_odom_msg_);
						}
						return;
					}
					T = ndt.getFinalTransformation();
					fitness = ndt.getFitnessScore();
				}
				const double dx = T(0,3);
				const double dy = T(1,3);
				const double yaw = std::atan2(T(1,0), T(0,0));
				
				//visualize_shit(filtered_target_pc.makeShared(), filtered_current_pc.makeShared(), result.makeShared());

				if (fitness > 0.1) // (std::hypot(dx, dy) > 0.25 || std::abs(yaw) > 0.35 || fitness > 0.05)
				{ 
					RCLCPP_WARN(this->get_logger(), "Rejecting ICP: dx=%.3f dy=%.3f yaw=%.3f fitness=%.5f", dx, dy, yaw, fitness);
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					if (!map_to_odom_msg_.header.frame_id.empty()) 
					{
						map_to_odom_msg_.header.stamp = msg->header.stamp;
						tf_broadcaster_->sendTransform(map_to_odom_msg_);
					}
					return;
				}
				// Publish markers to visualize result				
				//icp_publisher_->publish(marker_from_cloud(filtered_current_pc, result, msg, 1.0, 0.0, 1.0, 0)); // How much did ICP/NDT move the scans?

				// Final transform T is target <- source = prev <- curr
				tf2::Quaternion q;
				q.setRPY(0.0, 0.0, yaw);

				tf2::Transform T_icp(q,tf2::Vector3(dx, dy, 0.0));

				// 5. Update map pose estimate
				tf2::Transform T_map_prev_lidar = T_map_odom_ * (near_start_pose ? T_odom_first_lidar_ : T_odom_prev_lidar_);
				tf2::Transform T_map_curr_lidar = T_map_prev_lidar * T_icp;

				// 6. Compute map -> odom
				//T_map_odom_ = T_map_curr_lidar * T_odom_curr_lidar.inverse();
				
				T_map_odom_ = T_map_odom_ * T_icp.inverseTimes(tf2_guess); //update with the difference between guess and icp
				
				// 7. Broadcast map -> odom
				map_to_odom_msg_.header.stamp = current_stamp;
				map_to_odom_msg_.header.frame_id = "map";
				map_to_odom_msg_.child_frame_id = "odom";
				map_to_odom_msg_.transform.translation.x = T_map_odom_.getOrigin().x();
				map_to_odom_msg_.transform.translation.y = T_map_odom_.getOrigin().y();
				map_to_odom_msg_.transform.translation.z = T_map_odom_.getOrigin().z();
				map_to_odom_msg_.transform.rotation.x = T_map_odom_.getRotation().x();
				map_to_odom_msg_.transform.rotation.y = T_map_odom_.getRotation().y();
				map_to_odom_msg_.transform.rotation.z = T_map_odom_.getRotation().z();
				map_to_odom_msg_.transform.rotation.w = T_map_odom_.getRotation().w();
				tf_broadcaster_->sendTransform(map_to_odom_msg_);

				// 8. Store current as next reference
				filtered_prev_pc_ = filtered_current_pc;
				T_odom_prev_lidar_ = T_odom_curr_lidar;


				const auto runtime = this->get_clock()->now().nanoseconds() - start_time;

				RCLCPP_INFO(this->get_logger(), "-> Processed scan: %li ns", runtime);

				if (runtime > worst_time_)
				{
					worst_time_ = runtime;
				}

			};

			pc_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/deskewed_scan", 10);

			icp_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/icp_result", 2);

			tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
			
			tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

			tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

			scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/lidar/scan", 10, scan_callback);
			// Ensure we start with a valid TFs (for first iteration)
			T_odom_prev_lidar_.setIdentity();
			try
			{
				// Fetch the latest available map->odomTF
				T_map_odom_ = tf2_from_msg(this->tf_buffer_->lookupTransform("map", "odom", rclcpp::Time(), 1000ms));
			}
			catch(tf2::TransformException & ex)
			{
				RCLCPP_FATAL(this->get_logger(), "Could not get transform: %s-> Defaulting to identity.", ex.what());
				T_map_odom_.setIdentity();
			}

			imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>("/phidgets/imu/data_raw", 1, imu_callback);

			RCLCPP_INFO(this->get_logger(), "Localization node has been started.");
		}

		u_int32_t worst_time_{0};
	
		private:
		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
		std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
		rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;

		rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr icp_publisher_;
		rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pc_publisher_;

		bool prev_pc_exists_{false};
		pcl::PointCloud<pcl::PointXYZ> filtered_prev_pc_;

		pcl::PointCloud<pcl::PointXYZ> filtered_first_pc_;

		tf2::Transform T_odom_prev_lidar_;
		tf2::Transform T_odom_first_lidar_;
		tf2::Transform T_map_first_lidar_;
		tf2::Transform T_map_odom_;
		geometry_msgs::msg::TransformStamped map_to_odom_msg_;
		
		rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
		double ang_vel_{0};

		size_t counter_{0};
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<Localization>();
	rclcpp::spin(node);
	RCLCPP_INFO(node->get_logger(), "Worst processing time: %u ns", node->worst_time_);
	rclcpp::shutdown();
	return 0;
}
