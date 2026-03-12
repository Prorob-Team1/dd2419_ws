#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/clock.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <chrono>


using namespace std::chrono_literals;
constexpr auto TIMEOUT = 100ms; // pain

pcl::PointCloud<pcl::PointXYZ> scan_to_pc(const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
	/* 
	This function converts the laser scan readings into 2D point cloud in the lidar frame.
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
    ror.setRadiusSearch(0.50);
    ror.setMinNeighborsInRadius(2);
    ror.filter(filtered);

    return filtered;
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

visualization_msgs::msg::Marker marker_from_cloud(const pcl::PointCloud<pcl::PointXYZ> & current_cloud, const pcl::PointCloud<pcl::PointXYZ> & result_cloud, sensor_msgs::msg::LaserScan::SharedPtr msg) 
{
	/* 
	Creates a marker in the map frame for visualization purposes.
	*/
	visualization_msgs::msg::Marker marker;
	marker.header.frame_id = "lidar_link";
	marker.header.stamp = msg->header.stamp;
	marker.ns = "icp_debug";
	marker.id = 0;
	marker.type = visualization_msgs::msg::Marker::LINE_LIST;
	marker.action = visualization_msgs::msg::Marker::ADD;
	marker.scale.x = 0.01;
	marker.color.a = 1.0;
	marker.color.r = 1.0;
	marker.color.b = 1.0;

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

			auto scan_callback = [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) -> void 
			{	
				/*
				if (++counter_ < 5) 
				{
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					if (!map_to_odom_msg_.header.frame_id.empty()) 
					{
						map_to_odom_msg_.header.stamp = msg->header.stamp;
						tf_broadcaster_->sendTransform(map_to_odom_msg_);
					}
					return;
				}
				counter_ = 0;
				*/

				const long int start_time = this->get_clock()->now().nanoseconds();


				// 1. Convert scan to 2D points in lidar frame
				pcl::PointCloud<pcl::PointXYZ> current_pc = scan_to_pc(msg);
				pcl::PointCloud<pcl::PointXYZ> filtered_current_pc = filter_for_icp(current_pc);
				rclcpp::Time current_stamp = msg->header.stamp;

				if (filtered_current_pc.empty()) {
					RCLCPP_WARN(this->get_logger(), "Scan produced no valid points; skipping.");
					return;
				}

				if (this->prev_scan_msg_ == nullptr) 
				{
					// Store first scan
					prev_scan_msg_ = msg;
					prev_pc_ = current_pc;
					filtered_prev_pc_ = filter_for_icp(current_pc);
					filtered_first_pc_ = filtered_prev_pc_;
					try
					{
						T_odom_prev_base_ = tf2_from_msg(
							tf_buffer_->lookupTransform("odom", "base_link", msg->header.stamp, TIMEOUT)
						);
						T_map_first_base_ = tf2_from_msg(
							tf_buffer_->lookupTransform("map", "base_link", msg->header.stamp, TIMEOUT)
						);
					}
					catch(tf2::TransformException & ex)
					{
						RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					}
					RCLCPP_INFO(this->get_logger(), "No previous scan available, I do a nothing :))");
					return;
				}

				
				// 2. Get odom pose at current scan timestamp
				tf2::Transform T_odom_curr_base;
				try
				{
					T_odom_curr_base = tf2_from_msg(
						tf_buffer_->lookupTransform("odom", "base_link", msg->header.stamp, TIMEOUT)
					);
				}
				catch (tf2::TransformException & ex)
				{
					RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					return;
				}

				// 3. Relative motion prediction from odom
				// source=current, target=prev => guess must be prev <- curr
				tf2::Transform T_initial_guess = T_odom_prev_base_.inverseTimes(T_odom_curr_base);

				Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
				const tf2::Matrix3x3& R = T_initial_guess.getBasis();
				const tf2::Vector3& t = T_initial_guess.getOrigin();

				guess(0,0) = R[0][0]; guess(0,1) = R[0][1]; guess(0,2) = R[0][2];
				guess(1,0) = R[1][0]; guess(1,1) = R[1][1]; guess(1,2) = R[1][2];
				guess(2,0) = R[2][0]; guess(2,1) = R[2][1]; guess(2,2) = R[2][2];
				guess(0,3) = static_cast<float>(t.x());
				guess(1,3) = static_cast<float>(t.y());
				guess(2,3) = static_cast<float>(t.z());

				// 4. ICP
				pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

				auto icp_target_pc = filtered_prev_pc_; //(T_odom_curr_base.getOrigin().length() < 1.0 || false ) ? filtered_first_pc_ : filtered_prev_pc_;

				icp.setInputSource(filtered_current_pc.makeShared());
				icp.setInputTarget(icp_target_pc.makeShared());
				icp.setMaximumIterations(500);
				icp.setTransformationEpsilon(1e-8);
				icp.setEuclideanFitnessEpsilon(1e-8);
				icp.setMaxCorrespondenceDistance(1.0);

				pcl::PointCloud<pcl::PointXYZ> result_icp;
				icp.align(result_icp, guess);

				if (!icp.hasConverged()) {
					RCLCPP_WARN(this->get_logger(), "ICP did not converge (fitness=%f). Keeping previous map->odom.", icp.getFitnessScore());
					return;
				}

				// ICP transform is target <- source = prev <- curr
				Eigen::Matrix4f T = icp.getFinalTransformation();
				double yaw = std::atan2(T(1,0), T(0,0));

				tf2::Quaternion q;
				q.setRPY(0.0, 0.0, yaw);

				tf2::Transform T_icp(
					q,
					tf2::Vector3(T(0,3), T(1,3), 0.0)
				);

				// 5. Update map pose estimate
				tf2::Transform T_map_prev_base = T_map_odom_ * T_odom_prev_base_;
				tf2::Transform T_map_curr_base = T_map_prev_base * T_icp;

				// 6. Compute map -> odom
				T_map_odom_ = T_map_curr_base * T_odom_curr_base.inverse();
				
				// 7. Broadcast map -> odom
				map_to_odom_msg_.header.stamp = current_stamp;
				map_to_odom_msg_.header.frame_id = "odom";
				map_to_odom_msg_.child_frame_id = "map";
				map_to_odom_msg_.transform.translation.x = T_map_odom_.getOrigin().x();
				map_to_odom_msg_.transform.translation.y = T_map_odom_.getOrigin().y();
				map_to_odom_msg_.transform.translation.z = T_map_odom_.getOrigin().z();
				map_to_odom_msg_.transform.rotation.x = T_map_odom_.getRotation().x();
				map_to_odom_msg_.transform.rotation.y = T_map_odom_.getRotation().y();
				map_to_odom_msg_.transform.rotation.z = T_map_odom_.getRotation().z();
				map_to_odom_msg_.transform.rotation.w = T_map_odom_.getRotation().w();
				tf_broadcaster_->sendTransform(map_to_odom_msg_);

				// 8. Store current as next reference
				prev_scan_msg_ = msg;
				prev_pc_ = current_pc;
				filtered_prev_pc_ = filtered_current_pc;
				T_odom_prev_base_ = T_odom_curr_base;

				RCLCPP_INFO(this->get_logger(), "\t-> Processed scan: %li ns", this->get_clock()->now().nanoseconds() - start_time);

			};

			ipc_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/icp_result", 2);

			tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
			
			tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

			tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

			subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/lidar/scan", 10, scan_callback);
			// Ensure we start with a valid TFs (for first iteration)
			T_odom_prev_base_.setIdentity();
			try
			{
				// Fetch the latest available map->odomTF
				T_map_odom_ = tf2_from_msg(this->tf_buffer_->lookupTransform("map", "odom", rclcpp::Time(), 1000ms));
			}
			catch(tf2::TransformException & ex)
			{
				RCLCPP_FATAL(this->get_logger(), "Could not get transform: %s. Defaulting to identity.", ex.what());
				T_map_odom_.setIdentity();
			}

			RCLCPP_INFO(this->get_logger(), "Localization node has been started.");
		}
	private:
		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
		std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
		rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;

		rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ipc_publisher_;

		sensor_msgs::msg::LaserScan::SharedPtr prev_scan_msg_;
		pcl::PointCloud<pcl::PointXYZ> prev_pc_;
		pcl::PointCloud<pcl::PointXYZ> filtered_prev_pc_;

		pcl::PointCloud<pcl::PointXYZ> filtered_first_pc_;

		tf2::Transform T_odom_prev_base_;
		tf2::Transform T_map_first_base_;
		tf2::Transform T_map_odom_;
		geometry_msgs::msg::TransformStamped map_to_odom_msg_;

		size_t counter_{0};
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Localization>());
	rclcpp::shutdown();
	return 0;
}
