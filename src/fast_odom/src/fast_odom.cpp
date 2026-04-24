#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/clock.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/utils.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <robp_interfaces/msg/encoders.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <thread>

constexpr bool USE_IMU = true; // if we want to use the IMU or not 
constexpr double TICKS_PER_REV = 48 * 64;
constexpr double WHEEL_RADIUS = 0.04921 - 0.001;
constexpr double BASE = 0.3135;


class FastOdom : public rclcpp::Node
{
	public:
		void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

		void encoder_callback(const robp_interfaces::msg::Encoders::SharedPtr msg);
		
		void calibrate_imu(const sensor_msgs::msg::Imu::SharedPtr msg);

		const double get_yaw(const rclcpp::Time stamp);

		const std::pair<double, double> encoder_delta(const robp_interfaces::msg::Encoders::SharedPtr msg);

		void broadcast_transform(const rclcpp::Time stamp, const double x, const double y, const double yaw);
		void publish_path(const rclcpp::Time stamp, const double x, const double y, const double yaw);
		void publish_marker(const rclcpp::Time stamp);

		FastOdom() : Node("fast_odom")
		{

			imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
				"/phidgets/imu/data_raw", 
				10, 
				[this](const sensor_msgs::msg::Imu::SharedPtr msg){imu_callback(msg);}
			);
		
			encoder_subscription_ = this->create_subscription<robp_interfaces::msg::Encoders>(
				"/phidgets/motor/encoders", 
				10, 
				[this](const robp_interfaces::msg::Encoders::SharedPtr msg){encoder_callback(msg);}
			);
			
      		path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);

			tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

			RCLCPP_INFO(this->get_logger(), "Fast-Odom node has been started.");

		};
		

	private:
		std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
		rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
		rclcpp::Subscription<robp_interfaces::msg::Encoders>::SharedPtr encoder_subscription_;
		rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
		rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;

		// IMU stuff
		rclcpp::Time last_imu_stamp_{};
		double yaw_imu_{0};
		double yaw_bias_{0};
		std::deque<std::pair<double,rclcpp::Time>> yaw_queue_;
		bool calibrated_{false};
		double cum_vel_readings_{0};
		size_t reading_counter_{0};

		// Encoder stuff
		double last_encoder_left_{0};
		double last_encoder_right_{0};

		// 2D pose
		double x_{0.0};
		double y_{0.0};
		double yaw_{0.0};

		// Path
		nav_msgs::msg::Path path_{};
};

void FastOdom::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) 
{	
	if (!calibrated_) 
	{
		calibrate_imu(msg);
		return;
	}
	if (last_imu_stamp_.nanoseconds() == 0) 
	{
		last_imu_stamp_ = msg->header.stamp;
		return;
	}

	const double w_z = -(msg->angular_velocity.z - yaw_bias_);
	const rclcpp::Duration dt = rclcpp::Time(msg->header.stamp) - last_imu_stamp_;
	double yaw = yaw_imu_ + w_z * dt.nanoseconds() / 1e9;
	// Wrap angle
	yaw_imu_ = std::atan2(std::sin(yaw), std::cos(yaw));
	last_imu_stamp_ = msg->header.stamp;

	yaw_queue_.emplace_back(yaw_imu_,last_imu_stamp_);
	if (yaw_queue_.size() > 20) 
	{
		yaw_queue_.pop_front();
	}
}

void FastOdom::encoder_callback(const robp_interfaces::msg::Encoders::SharedPtr msg) 
{
	/*
	Takes encoder readings and updates the odometry.

	This function is called every time the encoders are updated (i.e., when a message is published on the '/motor/encoders' topic).

	Your task is to update the odometry based on the encoder data in 'msg'. You are allowed to add/change things outside this function.

	Keyword arguments:
	msg -- An encoders ROS message. To see more information about it
	run 'ros2 interface show robp_interfaces/msg/Encoders' in a terminal.
	*/
	const double ticks_per_rev = TICKS_PER_REV;
	const double wheel_radius = WHEEL_RADIUS;
	const double base = BASE;

	const std::pair<double, double> encoder_deltas = encoder_delta(msg);
	const double delta_ticks_left = encoder_deltas.first;
	const double delta_ticks_right = encoder_deltas.second;

	const double delta_phi_r = 2.0 * M_PI * (delta_ticks_right / ticks_per_rev); 
	const double delta_phi_l = 2.0 * M_PI * (delta_ticks_left / ticks_per_rev);
	const double D = 0.5 * wheel_radius * (delta_phi_r + delta_phi_l);

	x_ += D * cos(yaw_);
	y_ += D * sin(yaw_);

	if (!USE_IMU) 
	{
		yaw_ += wheel_radius * (delta_phi_r - delta_phi_l) / base;
		yaw_ = std::atan2(std::sin(yaw_), std::cos(yaw_));
	}
	else
	{
		yaw_ = get_yaw(msg->header.stamp);
	}

	auto stamp = msg->header.stamp;

	broadcast_transform(stamp, x_, y_, yaw_);
	publish_path(stamp, x_, y_, yaw_);
	publish_marker(stamp);
}

void FastOdom::calibrate_imu(const sensor_msgs::msg::Imu::SharedPtr msg) 
{
	if (reading_counter_ < 100) 
	{
		cum_vel_readings_ += msg->angular_velocity.z;
		++reading_counter_;
		return;
	}

	yaw_bias_ = cum_vel_readings_ / reading_counter_;
	calibrated_ = true;
	RCLCPP_INFO(this->get_logger(), "IMU (gyro) calibrated using %lu readings. Calculated bias: %.6f", reading_counter_, yaw_bias_);
}

const double FastOdom::get_yaw(const rclcpp::Time t_enc)
{
	if (yaw_queue_.empty())
	{
		RCLCPP_INFO(this->get_logger(), "No gyro readings available, returning last known yaw");
		return yaw_imu_;
	}
	double min_t_diff = DBL_MAX;
	double closest_reading = yaw_imu_;
	for (size_t i = 0; i<yaw_queue_.size(); ++i) 
	{
		const rclcpp::Time t_imu = yaw_queue_.at(i).second;
		const double t_diff = abs(t_imu.nanoseconds() - t_enc.nanoseconds());
		if (t_diff < min_t_diff) 
		{
			closest_reading = yaw_queue_.at(i).first;
			min_t_diff = t_diff;
		}

	}

	return closest_reading;
}

const std::pair<double, double> FastOdom::encoder_delta(const robp_interfaces::msg::Encoders::SharedPtr msg) 
{
	double delta_left, delta_right;
	if (last_encoder_left_ == 0.0 || last_encoder_right_ == 0.0)
	{
		delta_left = msg->delta_encoder_left;
		delta_right = msg->delta_encoder_right;
	}
	else
	{
		delta_left = msg->encoder_left - last_encoder_left_;
		delta_right = msg->encoder_right - last_encoder_right_;
	}
	last_encoder_left_ = msg->encoder_left;
	last_encoder_right_ = msg->encoder_right;

	return {static_cast<double>(delta_left), static_cast<double>(delta_right)};
}

void FastOdom::broadcast_transform(const rclcpp::Time stamp, const double x, const double y, const double yaw) 
{
	/*
	Takes a 2D pose and broadcasts it as a ROS transform.

	Broadcasts a 3D transform with z, roll, and pitch all zero.
	The transform is stamped with the current time and is between the frames 'odom' -> 'base_link'.

	Keyword arguments:
	stamp -- timestamp of the transform
	x -- x coordinate of the 2D pose
	y -- y coordinate of the 2D pose
	yaw -- yaw of the 2D pose (in radians)
	*/

	geometry_msgs::msg::TransformStamped t{};
	t.header.stamp = stamp;
	t.header.frame_id = "odom";
	t.child_frame_id = "base_link";

	// The robot only exists in 2D, thus we set x and y translation
	// coordinates and set the z coordinate to 0
	t.transform.translation.x = x;
	t.transform.translation.y = y;
	t.transform.translation.z = 0.0;

	// For the same reason, the robot can only rotate around one axis
	// and this why we set rotation in x and y to 0 and obtain
	// rotation in z axis from the message
	tf2::Quaternion q = tf2::Quaternion::createFromRPY(0.0, 0.0, yaw);
	t.transform.rotation.x = q.getX();
	t.transform.rotation.y = q.getY();
	t.transform.rotation.z = q.getZ();
	t.transform.rotation.w = q.getW();

	// Send the transformation
	tf_broadcaster_->sendTransform(t);

}

void FastOdom::publish_path(const rclcpp::Time stamp, const double x, const double y, const double yaw) 
{
	/*
	Takes a 2D pose appends it to the path and publishes the whole path.

	Keyword arguments:
	stamp -- timestamp of the transform
	x -- x coordinate of the 2D pose
	y -- y coordinate of the 2D pose
	yaw -- yaw of the 2D pose (in radians)
	*/

	path_.header.stamp = stamp;
	path_.header.frame_id = "odom";

	geometry_msgs::msg::PoseStamped pose{};
	pose.header = path_.header;

	pose.pose.position.x = x;
	pose.pose.position.y = y;
	pose.pose.position.z = 0.01;  // 1 cm up so it will be above ground level

	tf2::Quaternion q = tf2::Quaternion::createFromRPY(0.0, 0.0, yaw);
	pose.pose.orientation.x = q.getX();
	pose.pose.orientation.y = q.getY();
	pose.pose.orientation.z = q.getZ();
	pose.pose.orientation.w = q.getW();

	path_.poses.emplace_back(pose);

	path_publisher_->publish(path_);
}

void FastOdom::publish_marker(const rclcpp::Time stamp) 
{
	/*
	Published a giant gray marker to visualize the robot
	*/
	visualization_msgs::msg::Marker robot_marker;
	robot_marker.header.frame_id = "base_link";
	robot_marker.header.stamp = stamp;
	robot_marker.ns = "robot_box";
	robot_marker.id = 1337;
	robot_marker.type = visualization_msgs::msg::Marker::CUBE;
	robot_marker.action = visualization_msgs::msg::Marker::ADD;
	robot_marker.pose.position.x = -0.1;
	robot_marker.pose.position.y = 0;
	robot_marker.pose.position.z = 0.05;
	robot_marker.scale.x = 0.38;
	robot_marker.scale.y = 0.27;
	robot_marker.scale.z = 0.13;
	robot_marker.color.r = 0.75;
	robot_marker.color.g = 0.75;
	robot_marker.color.b = 0.75;
	robot_marker.color.a = 0.8;
	marker_publisher_->publish(robot_marker);
}

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<FastOdom>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
