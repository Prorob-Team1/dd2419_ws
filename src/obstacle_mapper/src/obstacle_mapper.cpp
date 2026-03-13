#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/clock.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/utils.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;
constexpr auto TIMEOUT = 10ms;
constexpr int8_t UNKNOWN_SPACE = -1;
constexpr int8_t OCCUPIED_SPACE = 1;
constexpr int8_t FREE_SPACE = 0;

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

// https://stackoverflow.com/questions/65169078/c-finding-the-argsort-of-a-vectorfloat
std::vector<int> VectorArgSort(std::vector<float> const & v) {
    /* Get the indices that gives a sorted vector v*/
	
    std::vector<int> retIndices(v.size());
    std::iota(retIndices.begin(), retIndices.end(), 0);

    std::stable_sort(
        retIndices.begin(), retIndices.end(),
        [&v](int i1, int i2) {return v[i1] > v[i2];});

    return retIndices;
}

std::vector<tf2::Vector3> scan_to_vec(sensor_msgs::msg::LaserScan::SharedPtr scan, const tf2::Transform & T_start, const tf2::Transform & T_end)
{
	/* 
	This function deskews and converts the laser scan readings into a a vector of points in lidar frame.
	*/
	const size_t num_ranges = scan->ranges.size();
	std::vector<tf2::Vector3> vec;
	vec.reserve(num_ranges);
	tf2::Transform T_i;
	tf2::Transform T_start_i;
	tf2::Vector3 p_i;
	tf2::Quaternion q_i;
	float r;
	tf2::Vector3 p_out;

	// Sort ranges in descending order
	const std::vector<int> indices = VectorArgSort(scan->ranges);
	std::vector<float> ranges;
	for (auto i : indices) 
	{
		ranges.emplace_back(scan->ranges.at(i));
	}


	// Convert to (x,y,z) coordinates
	for (size_t i = 0; i < num_ranges; ++i) 
	{
		float range = ranges.at(i);
		if (range >= scan->range_min && range <= scan->range_max) 
		{
			// Convert (range,angle) to (x,y,z)
			float angle = scan->angle_min + indices.at(i) * scan->angle_increment;
			float x = range * cos(angle);
			float y = range * sin(angle);
			float z = 0.0;
			
			// Interpolate lidar point TF
			r = std::clamp((i * scan->time_increment) / ((num_ranges - 1)*scan->time_increment), 0.0f, 1.0f);
			p_i = tf2::lerp(T_start.getOrigin(), T_end.getOrigin(), r);
			q_i = tf2::slerp(T_start.getRotation(), T_end.getRotation(), r);
			T_i.setOrigin(p_i);
			T_i.setRotation(q_i);

			// TF point back to start TF
			T_start_i = T_start.inverseTimes(T_i);
			p_out = T_start_i * tf2::Vector3(x,y,z);

			vec.emplace_back(p_out);
		}
	}
	return vec;
}

int pose_to_grid_idx(const float x, const float y, const nav_msgs::msg::OccupancyGrid & map) {
	
	int idx;

	// Do shit

	return idx;
}

std::vector<int> ray_trace(const tf2::Vector3 & start, const tf2::Vector3 & end, const nav_msgs::msg::OccupancyGrid & map) 
{
	std::vector<int> grid_indeces;
	// do raytracing :D


	return grid_indeces;
}


class ObstacleMapper : public rclcpp::Node
{
	public:
		ObstacleMapper() : Node("obstacle_mapper")
		{

			auto map_callback = [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) -> void
			{
				if (!first_map_recieved_) 
				{
					// Copy map
					map_ = *msg;
					// Fill map with unknown space
					std::fill(map_.data.begin(), map_.data.end(), UNKNOWN_SPACE);
					RCLCPP_INFO(this->get_logger(), "First map recieved.");
					first_map_recieved_ = true;
				}

			};

			auto scan_callback = [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) -> void 
			{	
				if (!first_map_recieved_)
				{
					RCLCPP_INFO(this->get_logger(), "No map available, skipping");
					return;
				}
				
				const auto start_time = this->get_clock()->now().nanoseconds();

				
				// 1. Retrieve TF
				tf2::Transform T_map_curr_lidar;
				try
				{
					T_map_curr_lidar = tf2_from_msg(
						tf_buffer_->lookupTransform("map", "lidar_link", msg->header.stamp, TIMEOUT)
					);
				}
				catch (tf2::TransformException & ex)
				{
					RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					prev_T_exists_ = false; // de-skewing requires the tf from the previous scan, otherwise it will fail
					return;
				}

				if (!prev_T_exists_)
				{
					T_map_prev_lidar_ = T_map_curr_lidar;
					prev_T_exists_ = true;
					return;
				}

				// 2. Process raw scan points
				const std::vector<tf2::Vector3> scan_vec = scan_to_vec(msg,T_map_prev_lidar_, T_map_curr_lidar);

				// 3. Ray-trace and populate map
				for (size_t i = 0; i < scan_vec.size(); ++i)
				{
					tf2::Vector3 vec_in_map_frame = T_map_prev_lidar_ * scan_vec.at(i);
					std::vector<int> grid_indeces = ray_trace(T_map_prev_lidar_.getOrigin(), vec_in_map_frame, map_);
					for (auto k : grid_indeces) {
						map_.data.at(k) = FREE_SPACE;
					}
					map_.data.at(pose_to_grid_idx(vec_in_map_frame.getX(), vec_in_map_frame.getY(), map_)) = OCCUPIED_SPACE;
				}

				// 4. Publish map
				map_.header.stamp = msg->header.stamp;
				map_publisher_->publish(map_);


				// 5. Report processing time.
				const auto runtime = this->get_clock()->now().nanoseconds() - start_time;

				RCLCPP_INFO(this->get_logger(), "-> Processed scan: %li ns", runtime);

				if (runtime > worst_time_)
				{
					worst_time_ = runtime;
				}

			};

			tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
			
			tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

			map_subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/occupancy_grid", 10, map_callback);

			map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/obstacle_map", 10);
		
			scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/lidar/scan", 10, scan_callback);

			RCLCPP_INFO(this->get_logger(), "ObstacleMapper node has been started.");

		}
		
		u_int32_t worst_time_{0};

	private:
		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
		std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
		rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscription_;
		rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
		rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;

		nav_msgs::msg::OccupancyGrid map_;
		tf2::Transform T_map_prev_lidar_;
		bool prev_T_exists_{false};
		bool first_map_recieved_{false};
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<ObstacleMapper>();
	rclcpp::spin(node);
	RCLCPP_INFO(node->get_logger(), "Worst processing time: %u ns", node->worst_time_);
	rclcpp::shutdown();
	return 0;
}