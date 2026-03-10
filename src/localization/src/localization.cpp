#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <chrono>


using namespace std::chrono_literals;
constexpr auto TIMEOUT = 1000ms; // pain

pcl::PointCloud<pcl::PointXYZ> scan_to_pc(const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
	/* 
	This function converts the laser scan readings into 2D points in the lidar frame.
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

class Localization : public rclcpp::Node
{
	public:
		Localization() : Node("localization")
		{
			auto scan_callback = [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) -> void 
			{	
				if (counter_++ < 10) { return; }

				counter_ = 0;
				const long int start_time = this->get_clock()->now().nanoseconds();


				// 1. Convert scan to 2D points in lidar frame
				pcl::PointCloud<pcl::PointXYZ> current_pc = scan_to_pc(msg);
				rclcpp::Time current_stamp = msg->header.stamp;

				if (this->prev_scan_msg_ == nullptr) 
				{
					prev_scan_msg_ = msg;
					prev_pc_ = current_pc;
					try
					{
						T_odom_prev_base_ = tf2_from_msg(tf_buffer_->lookupTransform("odom", "base_link", prev_scan_msg_->header.stamp, TIMEOUT));
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
				tf2::Transform T_initial_guess = T_odom_prev_base_.inverseTimes(T_odom_curr_base);
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
				

				// 4. ICP between prev/reference scan and current scan
				pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
				icp.setInputSource(current_pc.makeShared());
				icp.setInputTarget(prev_pc_.makeShared());
				icp.setMaximumIterations(10);
				icp.setTransformationEpsilon(1e-8);
				icp.setEuclideanFitnessEpsilon(1e-8);
				icp.setMaxCorrespondenceDistance(0.5);
				pcl::PointCloud<pcl::PointXYZ> result_icp;
				icp.align(result_icp, guess);


				// 5. Update map pose estimate (accumulate using ICP delta)
				Eigen::Matrix4f T = icp.getFinalTransformation();

				double x = T(0, 3);
				double y = T(1, 3);
				double yaw = std::atan2(T(1, 0), T(0, 0));

				tf2::Quaternion q;
				q.setRPY(0.0, 0.0, yaw);

				tf2::Transform T_icp(q, tf2::Vector3(x, y, 0.0));
				// PCL ICP returns a transform that maps current -> previous; use its inverse to get prev->curr
				tf2::Transform delta_icp = T_icp.inverse();
				T_map_base_ = T_map_base_ * delta_icp;

				// 6. Compute map -> odom
				// T_map_odom = T_map_base * inverse(T_odom_curr_base)
				tf2::Transform T_map_odom = T_map_base_ * T_odom_curr_base.inverse();

				// 7. Broadcast map -> odom
				geometry_msgs::msg::TransformStamped map_to_odom_msg;
				map_to_odom_msg.header.stamp = current_stamp;
				map_to_odom_msg.header.frame_id = "map";
				map_to_odom_msg.child_frame_id = "odom";
				map_to_odom_msg.transform.translation.x = T_map_odom.getOrigin().x();
				map_to_odom_msg.transform.translation.y = T_map_odom.getOrigin().y();
				map_to_odom_msg.transform.translation.z = T_map_odom.getOrigin().z();
				map_to_odom_msg.transform.rotation.x = T_map_odom.getRotation().x();
				map_to_odom_msg.transform.rotation.y = T_map_odom.getRotation().y();
				map_to_odom_msg.transform.rotation.z = T_map_odom.getRotation().z();
				map_to_odom_msg.transform.rotation.w = T_map_odom.getRotation().w();
				tf_broadcaster_->sendTransform(map_to_odom_msg);

				// 8. Store current as next reference
				prev_scan_msg_ = msg;
				prev_pc_ = current_pc;
				T_odom_prev_base_ = T_odom_curr_base;

				RCLCPP_INFO(this->get_logger(), "Processed scan: %li ns", this->get_clock()->now().nanoseconds() - start_time);

			};

			tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
			
			tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

			tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

			subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/lidar/scan", 10, scan_callback);

			RCLCPP_INFO(this->get_logger(), "Localization node has been started.");
		}
	private:
		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
		std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
		rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;

		sensor_msgs::msg::LaserScan::SharedPtr prev_scan_msg_;
		pcl::PointCloud<pcl::PointXYZ> prev_pc_;

		tf2::Transform T_odom_prev_base_;
		tf2::Transform T_map_base_;

		size_t counter_{0};
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Localization>());
	rclcpp::shutdown();
	return 0;
}
