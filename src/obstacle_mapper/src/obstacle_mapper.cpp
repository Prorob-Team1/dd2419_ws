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
#include <chrono>
#include <thread>

using namespace std::chrono_literals;
constexpr auto TIMEOUT = 100ms;
constexpr int8_t UNKNOWN_SPACE = -1;
constexpr int8_t OCCUPIED_SPACE = 1;
constexpr int8_t OCCUPIED_SPACE_MAX = 100;
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


struct GridIdx
{
	int x;
	int y;
};

GridIdx pose_to_grid_idx(const double x, const double y, const nav_msgs::msg::OccupancyGrid & map) {
	int x_idx = static_cast<int>(std::floor((x - map.info.origin.position.x)/map.info.resolution));
	int y_idx = static_cast<int>(std::floor((y - map.info.origin.position.y)/map.info.resolution));
	return GridIdx{x_idx, y_idx};
}

void ray_trace(const tf2::Vector3 & start, const tf2::Vector3 & end, nav_msgs::msg::OccupancyGrid & map) 
{
	/*
	A Fast Voxel Traversal Algorithm for Ray Tracing
	J. Amanatides, A. Woo
	http://www.cse.yorku.ca/~amana/research/grid.pdf
	*/

	const tf2::Vector3 v = end - start;
	const double res = map.info.resolution;
	const double origin_x = map.info.origin.position.x;
	const double origin_y = map.info.origin.position.y;

	const int step_x = (v.getX() > 0) ? 1 : (v.getX() < 0 ? -1 : 0);
	const int step_y = (v.getY() > 0) ? 1 : (v.getY() < 0 ? -1 : 0);

	const GridIdx current_idx = pose_to_grid_idx(start.getX(), start.getY(), map);
	const GridIdx end_idx = pose_to_grid_idx(end.getX(), end.getY(), map);

	int x = current_idx.x;
	int y = current_idx.y;
	const int x_end = end_idx.x;
	const int y_end = end_idx.y;

	const double next_boundary_x = (step_x > 0) ? (origin_x + (x + 1) * res) : (origin_x + x * res);

	const double next_boundary_y = (step_y > 0) ? (origin_y + (y + 1) * res) : (origin_y + y * res);

	double t_max_x = (step_x != 0) ? (next_boundary_x - start.getX()) / v.getX() : DBL_MAX;
	double t_max_y = (step_y != 0) ? (next_boundary_y - start.getY()) / v.getY() : DBL_MAX;

	const double t_delta_x = (step_x != 0) ? res / std::abs(v.getX()) : DBL_MAX;
	const double t_delta_y = (step_y != 0) ? res / std::abs(v.getY()) : DBL_MAX;

	while (x != x_end || y != y_end)
	{
		if (t_max_x < t_max_y)
		{
			x += step_x;
			t_max_x += t_delta_x;
		}
		else if (t_max_y < t_max_x)
		{
			y += step_y;
			t_max_y += t_delta_y;
		}
		else // edge case, traverse in both directions
		{
			x += step_x;
			y += step_y;
			t_max_x += t_delta_x;
			t_max_y += t_delta_y;
		}

		if (x >= 0 && x < static_cast<int>(map.info.width) && y >= 0 && y < static_cast<int>(map.info.height))
		{	
			if (x != x_end && y != y_end)
			{
				int8_t & cell = map.data.at(x + static_cast<int>(map.info.width) * y);
				cell = (cell <= FREE_SPACE || cell == UNKNOWN_SPACE) ? FREE_SPACE : --cell;
			}
		}
		else
		{
			break;
		}
	}
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
					// ### THIS IS THE MAJOR BOTTLENECK! ###
					T_map_curr_lidar = tf2_from_msg(
						tf_buffer_->lookupTransform("map", "lidar_link", msg->header.stamp+rclcpp::Duration::from_seconds(msg->scan_time), TIMEOUT) // We retrieve the TF for when the current scan *ends* (at least according to the source code of the lidar node)
					);
				}
				catch (tf2::TransformException & ex)
				{
					RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
					prev_T_exists_ = false; // de-skewing requires the tf from the previous scan (i.e. the TF from when the current scan started), otherwise it will fail
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
					// Raytrace to fill map with FREE_SPACE
					ray_trace(T_map_prev_lidar_.getOrigin(), vec_in_map_frame, map_);
					// Set end-point to OCCUPIED_SPACE
					const GridIdx idx = pose_to_grid_idx(vec_in_map_frame.getX(), vec_in_map_frame.getY(), map_);
					if (idx.x >= 0 && idx.x < static_cast<int>(map_.info.width) && idx.y >= 0 && idx.y < static_cast<int>(map_.info.height))
					{
						int8_t & cell = map_.data.at(idx.x + static_cast<int>(map_.info.width) * idx.y);
						cell = (cell < OCCUPIED_SPACE_MAX) ? ++cell : OCCUPIED_SPACE_MAX;
					}
				}

				// 4. Publish map
				map_.header.stamp = msg->header.stamp;
				map_publisher_->publish(map_);
				
				// 5. Store previous TF
				T_map_prev_lidar_ = T_map_curr_lidar;

				// 6. Report processing time.
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