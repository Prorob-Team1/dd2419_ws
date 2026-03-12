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
#include <chrono>


using namespace std::chrono_literals;
constexpr auto TIMEOUT = 100ms; // pain
constexpr bool DO_ICP = true;

pcl::PointCloud<pcl::PointXYZ> scan_to_pc(const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
	/* 
	This function converts the laser scan readings into a 2D point cloud in the lidar frame.
	*/
	const size_t num_ranges = scan->ranges.size();
	pcl::PointCloud<pcl::PointXYZ> pointcloud;
	for (size_t i = 0; i < num_ranges; ++i) {
		float range = scan->ranges[i];
		if (range >= scan->range_min && range <= scan->range_max) {
			float angle = scan->angle_min + i * scan->angle_increment;
			float x = range * cos(angle);
			float y = range * sin(angle);
			float z = 0.0;
			pointcloud.emplace_back(x, y, z);
		}
	}
	return pointcloud;
}

pcl::PointCloud<pcl::PointXYZ> filter_for_icp(const pcl::PointCloud<pcl::PointXYZ> & input)
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
	icp.setMaxCorrespondenceDistance(0.5);

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

static Eigen::Matrix4f guess_from_TFs(const tf2::Transform & T_prev, const tf2::Transform & T_curr) 
{
	tf2::Transform T_initial_guess = T_prev.inverseTimes(T_curr);

	Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
	const tf2::Matrix3x3& R = T_initial_guess.getBasis();
	const tf2::Vector3& t = T_initial_guess.getOrigin();

	guess(0,0) = R[0][0]; guess(0,1) = R[0][1]; guess(0,2) = R[0][2];
	guess(1,0) = R[1][0]; guess(1,1) = R[1][1]; guess(1,2) = R[1][2];
	guess(2,0) = R[2][0]; guess(2,1) = R[2][1]; guess(2,2) = R[2][2];
	guess(0,3) = static_cast<float>(t.x());
	guess(1,3) = static_cast<float>(t.y());
	guess(2,3) = static_cast<float>(t.z());

	return guess;
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
			auto loop_closure = [this]() -> tf2::Transform {
				
			};

			auto imu_callback = [this](const sensor_msgs::msg::Imu::SharedPtr msg) -> void
			{
				ang_vel_ = msg->angular_velocity.z;
			};

			auto scan_callback = [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) -> void 
			{	

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

				const auto start_time = this->get_clock()->now().nanoseconds();

				// 1. Convert scan to 2D points in lidar frame
				pcl::PointCloud<pcl::PointXYZ> current_pc = scan_to_pc(msg);
				pcl::PointCloud<pcl::PointXYZ> filtered_current_pc = filter_for_icp(current_pc);
				rclcpp::Time current_stamp = msg->header.stamp;

				if (filtered_current_pc.empty()) {
					RCLCPP_WARN(this->get_logger(), "Scan produced no valid points; skipping.");
					return;
				}

				if (!prev_pc_exists_) 
				{
					// Store first scan
					try
					{
						T_odom_prev_lidar_ = tf2_from_msg(
							tf_buffer_->lookupTransform("odom", "lidar_link", msg->header.stamp, TIMEOUT)
						);
						filtered_prev_pc_ = filtered_current_pc;
						filtered_first_pc_ = filtered_current_pc;
						T_odom_first_lidar_ = T_odom_prev_lidar_;
						prev_pc_exists_ = true;
					}
					catch(tf2::TransformException & ex)
					{
						RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
						return;
					}
					RCLCPP_INFO(this->get_logger(), "No previous scan available, cannot run ICP");
				}
				
				// 2. Get odom pose at current scan timestamp
				tf2::Transform T_odom_curr_base;
				try
				{
					T_odom_curr_base = tf2_from_msg(
						tf_buffer_->lookupTransform("odom", "lidar_link", msg->header.stamp, TIMEOUT)
					);
				}
				catch (tf2::TransformException & ex)
				{
					RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					return;
				}

				// 3. Relative motion prediction from odom
				// source=current, target=prev => guess must be prev <- curr
				
				float rot_diff = static_cast<float>(tf2::getYaw(T_odom_curr_base.getRotation()) - tf2::getYaw(T_odom_first_lidar_.getRotation()));
				rot_diff = std::abs(std::atan2(sin(rot_diff), cos(rot_diff))); // wrap it to [0,pi]!
				const float dist_to_start = static_cast<float>(T_odom_curr_base.getOrigin().distance(T_odom_first_lidar_.getOrigin()));
				const bool near_start_pose = false;// dist_to_start < 0.3 && rot_diff < 0.3;
				const Eigen::Matrix4f guess = guess_from_TFs(near_start_pose ? T_odom_first_lidar_ : T_odom_prev_lidar_, T_odom_curr_base);
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
						return;
					}
					T = ndt.getFinalTransformation();
					fitness = ndt.getFitnessScore();
				}
				const double dx = T(0,3);
				const double dy = T(1,3);
				const double yaw = std::atan2(T(1,0), T(0,0));
				

				if (fitness > 0.1) // (std::hypot(dx, dy) > 0.25 || std::abs(yaw) > 0.35 || fitness > 0.05)
				{ 
					RCLCPP_WARN(this->get_logger(), "Rejecting ICP: dx=%.3f dy=%.3f yaw=%.3f fitness=%.5f", dx, dy, yaw, fitness);
					return;
				}
				// Publish markers to visualize result
				icp_publisher_->publish(marker_from_cloud(filtered_current_pc, result, msg, 1.0, 0.0, 1.0, 0)); // How much did ICP/NDT move the scans?

				// Final transform T is target <- source = prev <- curr
				tf2::Quaternion q;
				q.setRPY(0.0, 0.0, yaw);

				tf2::Transform T_icp(q,tf2::Vector3(dx, dy, 0.0));

				// 5. Update map pose estimate
				tf2::Transform T_map_prev_base = T_map_odom_ * (near_start_pose ? T_odom_first_lidar_ : T_odom_prev_lidar_);
				tf2::Transform T_map_curr_base = T_map_prev_base * T_icp;

				// 6. Compute map -> odom
				T_map_odom_ = T_map_curr_base * T_odom_curr_base.inverse();
				
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
				T_odom_prev_lidar_ = T_odom_curr_base;

				RCLCPP_INFO(this->get_logger(), "-> Processed scan: %li ns", this->get_clock()->now().nanoseconds() - start_time);

			};

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
	private:
		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
		std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
		rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;

		rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr icp_publisher_;

		bool prev_pc_exists_{false};
		pcl::PointCloud<pcl::PointXYZ> filtered_prev_pc_;

		pcl::PointCloud<pcl::PointXYZ> filtered_first_pc_;

		tf2::Transform T_odom_prev_lidar_;
		tf2::Transform T_odom_first_lidar_;
		tf2::Transform T_map_odom_;
		geometry_msgs::msg::TransformStamped map_to_odom_msg_;
		
		rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
		double ang_vel_{0};

		size_t counter_{0};
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Localization>());
	rclcpp::shutdown();
	return 0;
}
