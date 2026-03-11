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

pcl::PointCloud<pcl::PointXYZ> filter_for_icp(const pcl::PointCloud<pcl::PointXYZ>& input)
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
			auto scan_callback = [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) -> void 
			{	
				if (counter_++ < 5) 
				{
					// Broadcast old map -> odom TF every time (unless it's time to update it)
					geometry_msgs::msg::TransformStamped map_to_odom_msg;
					map_to_odom_msg.header.stamp = msg->header.stamp;
					map_to_odom_msg.header.frame_id = "map";
					map_to_odom_msg.child_frame_id = "odom";
					map_to_odom_msg.transform.translation.x = T_map_odom_.getOrigin().x();
					map_to_odom_msg.transform.translation.y = T_map_odom_.getOrigin().y();
					map_to_odom_msg.transform.translation.z = T_map_odom_.getOrigin().z();
					map_to_odom_msg.transform.rotation.x = T_map_odom_.getRotation().x();
					map_to_odom_msg.transform.rotation.y = T_map_odom_.getRotation().y();
					map_to_odom_msg.transform.rotation.z = T_map_odom_.getRotation().z();
					map_to_odom_msg.transform.rotation.w = T_map_odom_.getRotation().w();
					tf_broadcaster_->sendTransform(map_to_odom_msg);
					return;
				}

				counter_ = 0;
				const long int start_time = this->get_clock()->now().nanoseconds();


				// 1. Convert scan to 2D points in lidar frame
				pcl::PointCloud<pcl::PointXYZ> current_pc = scan_to_pc(msg);
				rclcpp::Time current_stamp = msg->header.stamp;

				if (current_pc.empty()) {
					RCLCPP_WARN(this->get_logger(), "Scan produced no valid points; skipping.");
					return;
				}

				if (this->prev_scan_msg_ == nullptr) 
				{
					prev_scan_msg_ = msg;
					prev_pc_ = current_pc;
					filtered_prev_pc_ = filter_for_icp(current_pc);
					try
					{
						T_odom_prev_lidar_ = tf2_from_msg(
							tf_buffer_->lookupTransform("lidar_link", "odom", msg->header.stamp, TIMEOUT)
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
				tf2::Transform T_odom_curr_lidar;

				try 
				{
					T_odom_curr_lidar = tf2_from_msg(
						tf_buffer_->lookupTransform("lidar_link", "odom", msg->header.stamp, TIMEOUT)
					);
				} 
				catch (tf2::TransformException & ex)
				{
					RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					return;
				}


				// 3. Relative motion prediction from odom
				tf2::Transform T_initial_guess = T_odom_curr_lidar *T_odom_prev_lidar_.inverse();
				Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
				const tf2::Matrix3x3& R = T_initial_guess.getBasis();
				const tf2::Vector3& t = T_initial_guess.getOrigin();
				// Rotation
				guess(0,0) = R[0][0]; guess(0,1) = R[0][1]; guess(0,2) = R[0][2];
				guess(1,0) = R[1][0]; guess(1,1) = R[1][1]; guess(1,2) = R[1][2];
				guess(2,0) = R[2][0]; guess(2,1) = R[2][1]; guess(2,2) = R[2][2];
				// Translation
				guess(0,3) = static_cast<float>(t.x());
				guess(1,3) = static_cast<float>(t.y());
				guess(2,3) = static_cast<float>(t.z());
				

				// 4. ICP between prev scan and current scan
				// TODO: Make pc's shared ptrs from the getgo, this heap allocation should only be done once per scan
				pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
				auto filtered_current_pc = filter_for_icp(current_pc);
				if (filtered_current_pc.empty()) 
				{
					RCLCPP_FATAL(this->get_logger(), "AAAAAAAAAAAAAAAAAAAAAaaa");
				}
				if (filtered_prev_pc_.empty()) 
				{
					RCLCPP_FATAL(this->get_logger(), "BBBBBBBBBBBAaaaaaaaaaaaa");
				}
				icp.setInputSource(filtered_current_pc.makeShared());
				icp.setInputTarget(filtered_prev_pc_.makeShared());
				icp.setMaximumIterations(50);
				icp.setTransformationEpsilon(1e-8);
				icp.setEuclideanFitnessEpsilon(1e-8);
				icp.setMaxCorrespondenceDistance(0.5);
				pcl::PointCloud<pcl::PointXYZ> result_icp;
				icp.align(result_icp, guess);


				// If ICP fails, keep the previous map->base boundary (do not drift to identity)
				if (!icp.hasConverged()) {
					RCLCPP_WARN(this->get_logger(), "ICP did not converge (fitness=%f). Keeping previous map->base.", icp.getFitnessScore());
					return;
				}
				
				// DEBUG: publish markers to see icp result:
				this->ipc_publisher_->publish(marker_from_cloud(filtered_current_pc, result_icp, msg));

				// 5. Update map pose estimate (accumulate using ICP delta)
				Eigen::Matrix4f T = icp.hasConverged() ? icp.getFinalTransformation() : Eigen::Matrix4f::Identity();

				float x = T(0, 3);
				float y = T(1, 3);
				float yaw = std::atan2(T(1, 0), T(0, 0));

				RCLCPP_INFO(get_logger(), "\n\t* ICP dx=%.3f dy=%.3f yaw=%.3f rad", x, y, yaw);
				RCLCPP_INFO(get_logger(), "\t* ICP converged: %s", icp.hasConverged() ? "true" : "false");
				RCLCPP_INFO(get_logger(), "\t* ICP fitness: %.6f", icp.getFitnessScore());

				tf2::Quaternion q;
				q.setRPY(0.0, 0.0, yaw);

				tf2::Transform T_icp(q, tf2::Vector3(x, y, 0.0));
				tf2::Transform T_map_lidar = T_icp.inverseTimes(T_odom_curr_lidar) * T_map_odom_;

				// 6. Compute map -> odom
				T_map_odom_ = T_map_lidar * T_odom_curr_lidar.inverse();
				
				// 7. Broadcast map -> odom
				geometry_msgs::msg::TransformStamped map_to_odom_msg;
				map_to_odom_msg.header.stamp = current_stamp;
				map_to_odom_msg.header.frame_id = "map";
				map_to_odom_msg.child_frame_id = "odom";
				map_to_odom_msg.transform.translation.x = T_map_odom_.getOrigin().x();
				map_to_odom_msg.transform.translation.y = T_map_odom_.getOrigin().y();
				map_to_odom_msg.transform.translation.z = T_map_odom_.getOrigin().z();
				map_to_odom_msg.transform.rotation.x = T_map_odom_.getRotation().x();
				map_to_odom_msg.transform.rotation.y = T_map_odom_.getRotation().y();
				map_to_odom_msg.transform.rotation.z = T_map_odom_.getRotation().z();
				map_to_odom_msg.transform.rotation.w = T_map_odom_.getRotation().w();
				tf_broadcaster_->sendTransform(map_to_odom_msg);

				// 8. Store current as next reference
				prev_scan_msg_ = msg;
				prev_pc_ = current_pc;
				filtered_prev_pc_ = filtered_current_pc;
				T_odom_prev_lidar_ = T_odom_curr_lidar;

				RCLCPP_INFO(this->get_logger(), "\t-> Processed scan: %li ns", this->get_clock()->now().nanoseconds() - start_time);

			};

			ipc_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/icp_result", 2);

			tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
			
			tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

			tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

			subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/lidar/scan", 10, scan_callback);
			// Ensure we start with a valid TFs (for first iteration)
			T_odom_prev_lidar_.setIdentity();
			try
			{
				// Fetch the latest available map->odomTF
				T_map_odom_ = tf2_from_msg(this->tf_buffer_->lookupTransform("odom", "map", rclcpp::Time(), 1000ms));
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

		tf2::Transform T_odom_prev_lidar_;
		tf2::Transform T_map_odom_;

		size_t counter_{0};
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Localization>());
	rclcpp::shutdown();
	return 0;
}
