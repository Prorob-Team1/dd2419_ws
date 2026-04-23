#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/filters/radius_outlier_removal.h>
#include <sensor_msgs/msg/imu.hpp>
#include <robp_interfaces/msg/duty_cycles.hpp>


using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class Localization : public rclcpp::Node
{
public:
    Localization() : Node("localization"), is_initialized_(false),
                     T_map_odom_(Eigen::Matrix4f::Identity())
    {
		this->declare_parameter<bool>("use_icp", true);
		this->get_parameter("use_icp", use_icp_);

        tf_buffer_      = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

		rclcpp::SubscriptionOptions scan_subscription_options;
		scan_subscription_options.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);;

        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/lidar/scan", 10,
            std::bind(&Localization::scanCallback, this, std::placeholders::_1), 
			scan_subscription_options);
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
			"/phidgets/imu/data_raw", 1, 
			std::bind(&Localization::imuCallback, this, std::placeholders::_1));

		duty_cycle_subscription_ = this->create_subscription<robp_interfaces::msg::DutyCycles>(
			"/phidgets/motor/duty_cycles", 1,
			std::bind(&Localization::dutyCycleCallback, this, std::placeholders::_1));


		rclcpp::QoS qos(rclcpp::KeepLast(1));
		qos.reliable();
		qos.transient_local();
        initial_pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initial_pose", qos,
            std::bind(&Localization::initialPoseCallback, this, std::placeholders::_1));

        tf_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&Localization::publishMapToOdom, this));

		pub_ref_cloud_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("/icp/reference", 1);
		pub_scan_guess_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/icp/guess", 1);
		pub_scan_aligned_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/icp/aligned", 1);


        RCLCPP_INFO(this->get_logger(), "Localization node started. Waiting for initial pose on /initialpose...");
    }

private:
	// node parameters
	bool use_icp_;

	// pubs and subs
    std::unique_ptr<tf2_ros::Buffer>               tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>    tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ref_cloud_;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_guess_;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_aligned_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
	rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscription_;
	rclcpp::Subscription<robp_interfaces::msg::DutyCycles>::SharedPtr duty_cycle_subscription_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::TimerBase::SharedPtr init_retry_timer_;  // fires until TF becomes available

	// state
    bool            is_initialized_;
    CloudT::Ptr     reference_cloud_;
    Eigen::Matrix4f T_map_lidar_0_;  // true lidar pose in map at t=0
    Eigen::Matrix4f T_map_odom_;     // the correction we own and publish
	Eigen::Matrix4f T_odom_base_;
	Eigen::Matrix4f T_odom_base_last_ = Eigen::Matrix4f::Identity();

	// save the last time stamp we took a scan
    rclcpp::Time last_scan_time_;

	rclcpp::Time last_driving_time_ = this->get_clock()->now() - rclcpp::Duration::from_seconds(1000);
	bool is_stationary_ = true;
	float distance_travelled_ = 0.0f;
	float distance_travelled_last_icp_ = 0.0f;

    // Stored so the retry timer can use it
    geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pending_initial_pose_;

	bool has_reference_cloud_ = false;
	Eigen::Matrix4f T_base_lidar_ = Eigen::Matrix4f::Identity(); 

	int reference_scan_count_ = 0;
	const int kReferenceScanTarget = 5;
    
    const float icpFitnessThreshold = 0.1f;
    const float maxAngularVelForGoodScan = 0.3f;
    const float icpInterpolationAlpha = 0.05f;
    const float icpCorrectionDistanceFromStart = 1.5f;
	const float stationaryTimeThreshold = 1.0f;
	const float accurateOdomDistance = 7.5f;


	double ang_vel_{0};



    void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        if (is_initialized_) return;

        pending_initial_pose_ = msg;
        tryInitialize();
    }

    void tryInitialize()
    {
        // Manually convert PoseWithCovarianceStamped -> Eigen (T_map_base)
        const auto & pos = pending_initial_pose_->pose.pose.position;
        const auto & q   = pending_initial_pose_->pose.pose.orientation;
        Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
        Eigen::Matrix4f T_map_base = Eigen::Matrix4f::Identity();
        T_map_base.block<3,3>(0,0) = quat.toRotationMatrix();
        T_map_base(0,3) = pos.x;
        T_map_base(1,3) = pos.y;
        T_map_base(2,3) = pos.z;

        // Get base_link -> lidar_link (static, won't change)
        Eigen::Matrix4f T_base_lidar;
        try {
            auto tf_base_lidar = tf_buffer_->lookupTransform(
                "base_link", "lidar_link", tf2::TimePointZero);
            T_base_lidar = transformToMatrix(tf_base_lidar);
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN(this->get_logger(), "Waiting for base_link->lidar_link: %s. Retrying...", e.what());
            scheduleRetry();
            return;
        }

        // T_map_lidar_0 = T_map_base * T_base_lidar
        T_map_lidar_0_ = T_map_base * T_base_lidar;

        // T_map_odom = T_map_base
        T_map_odom_ = T_map_base;
		T_base_lidar_ = T_base_lidar;

        is_initialized_ = true;
        if (init_retry_timer_) {
            init_retry_timer_->cancel();
            init_retry_timer_ = nullptr;
        }

        RCLCPP_INFO(this->get_logger(), "Initialization complete.");
    }

    void scheduleRetry()
    {
        if (init_retry_timer_) return;  // already scheduled
        init_retry_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            [this]() {
                tryInitialize();
            });
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
		ang_vel_ = msg->angular_velocity.z;
	}

	void dutyCycleCallback(const robp_interfaces::msg::DutyCycles::SharedPtr msg) {
		bool isDriving = std::abs(msg->duty_cycle_left) > 0.01 || std::abs(msg->duty_cycle_right) > 0.01;
		auto now = this->get_clock()->now();
		if (isDriving) last_driving_time_ = now;
		is_stationary_ = last_driving_time_ + rclcpp::Duration::from_seconds(stationaryTimeThreshold) < now;
	}

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
	{
		if (!use_icp_) return;
		if (!is_initialized_) {
			RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
								"Not initialized yet, skipping scan.");
			return;
		}

		// Step 1: Convert LaserScan -> PCL cloud (in lidar frame)
		CloudT::Ptr scan_cloud = laserScanToCloud(msg);
		if (scan_cloud->empty()) {
			RCLCPP_WARN(this->get_logger(), "Empty scan, skipping.");
			return;
		}

		// Step 2: First scan after init -> transform into map frame, store as reference
		if (!has_reference_cloud_) {
			if (!reference_cloud_) {
				reference_cloud_ = CloudT::Ptr(new CloudT());
			}

			CloudT transformed;
			pcl::transformPointCloud(*scan_cloud, transformed, T_map_lidar_0_);
			*reference_cloud_ += transformed;  // append points
			reference_scan_count_++;

			if (reference_scan_count_ >= kReferenceScanTarget) {
				// filter the reference cloud to remove outliers
				reference_cloud_ = filterCloud(reference_cloud_);
				has_reference_cloud_ = true;
				RCLCPP_INFO(this->get_logger(), "Reference cloud ready with %zu points.", reference_cloud_->size());
			}
			return;
		}

		publishCloud(reference_cloud_, "map", pub_ref_cloud_);


		// Step 3: Look up current odom-based lidar pose as ICP initial guess
		try {
			auto tf_odom_base = tf_buffer_->lookupTransform(
				"odom", "base_link", tf2::TimePointZero);
			T_odom_base_ = transformToMatrix(tf_odom_base);
			addDistanceTraveled();
			T_odom_base_last_ = T_odom_base_;
		} catch (const tf2::TransformException & e) {
			RCLCPP_WARN(this->get_logger(), "Could not get odom->base_link: %s", e.what());
			return;
		}


		// T_map_lidar_1_odom = T_map_odom * T_odom_base * T_base_lidar
		Eigen::Matrix4f T_map_lidar_1_guess = T_map_odom_ * T_odom_base_ * T_base_lidar_;

		// if the robot is far away from the reference pose, then skip ICP
		float initial_distance = (T_map_lidar_1_guess.block<3,1>(0,3) - T_map_lidar_0_.block<3,1>(0,3)).norm();
		if (initial_distance > icpCorrectionDistanceFromStart) return;  // tune this threshold for your environment
			
		if (std::abs(ang_vel_) > maxAngularVelForGoodScan) return;

		CloudT::Ptr cloud_at_guess(new CloudT());
		pcl::transformPointCloud(*scan_cloud, *cloud_at_guess, T_map_lidar_1_guess);
		scan_cloud = filterCloud(scan_cloud);

		publishCloud(cloud_at_guess, "map", pub_scan_guess_);

		// Step 4: Run ICP
		// Target: reference cloud (in map frame)
		// Source: current scan (in lidar frame), seeded with initial guess
		pcl::IterativeClosestPoint<PointT, PointT> icp;
		icp.setInputTarget(reference_cloud_);
		icp.setInputSource(scan_cloud);
		icp.setMaximumIterations(50);
		icp.setTransformationEpsilon(1e-6);
		icp.setMaxCorrespondenceDistance(0.4);  // tune this for your environment

		CloudT aligned;
		icp.align(aligned, T_map_lidar_1_guess);

		if (!icp.hasConverged()) {
			RCLCPP_WARN(this->get_logger(), "ICP did not converge, skipping update.");
			return;
		}

        const float score = icp.getFitnessScore();
        if (score > icpFitnessThreshold) {
				// RCLCPP_WARN(this->get_logger(),
				// 			"ICP fitness %.4f > threshold %.4f, skipping update.",
				// 			score, kIcpFitnessThreshold);
				return;
        }

		// RCLCPP_DEBUG(this->get_logger(), "ICP converged, score: %.4f", icp.getFitnessScore());

		// Step 5: Update T_map_odom
		Eigen::Matrix4f T_map_lidar_1_icp = icp.getFinalTransformation();

		Eigen::Matrix4f T_map_base_1_icp = T_map_lidar_1_icp * T_base_lidar_.inverse();

		Eigen::Matrix4f T_map_odom_icp = T_map_base_1_icp * T_odom_base_.inverse();
		T_map_odom_ = interpolateTransform(T_map_odom_, T_map_odom_icp, icpInterpolationAlpha);

		CloudT::Ptr aligned_ptr(new CloudT(aligned));
		publishCloud(aligned_ptr, "map", pub_scan_aligned_);

	}

	CloudT::Ptr filterCloud(const CloudT::Ptr & input)
	{
		CloudT::Ptr filtered(new CloudT());
		pcl::RadiusOutlierRemoval<PointT> ror;
		ror.setInputCloud(input);       // takes shared_ptr directly, no makeShared() needed
		ror.setRadiusSearch(0.20);
		ror.setMinNeighborsInRadius(2);
		ror.filter(*filtered);
		return filtered;                // returns shared_ptr, no copy of point data
	}

	Eigen::Matrix4f interpolateTransform(const Eigen::Matrix4f & A,
                                      const Eigen::Matrix4f & B,
                                      float alpha)
	{
		// Translation: lerp
		Eigen::Vector3f t = (1.f - alpha) * A.block<3,1>(0,3) + alpha * B.block<3,1>(0,3);

		// Rotation: slerp
		Eigen::Quaternionf qa(A.block<3,3>(0,0));
		Eigen::Quaternionf qb(B.block<3,3>(0,0));
        // ! use lower alpha quaternion interpolation
		Eigen::Quaternionf q = qa.slerp(alpha / 10, qb);

		Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
		result.block<3,3>(0,0) = q.toRotationMatrix();
		result.block<3,1>(0,3) = t;
		return result;
	}

    void publishMapToOdom()
    {
        if (!is_initialized_) return;

        auto tf_msg = matrixToTransform(T_map_odom_);
		tf_msg.header.stamp    = this->get_clock()->now();
		tf_msg.header.frame_id = "map";
		tf_msg.child_frame_id  = "odom";
		tf_broadcaster_->sendTransform(tf_msg);

    }

	CloudT::Ptr laserScanToCloud(const sensor_msgs::msg::LaserScan::SharedPtr msg)
	{
		CloudT::Ptr cloud(new CloudT());
		float angle = msg->angle_min;
		for (size_t i = 0; i < msg->ranges.size(); ++i, angle += msg->angle_increment) {
			float r = msg->ranges[i];
			if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) continue;
			PointT p;
			p.x = r * std::cos(angle);
			p.y = r * std::sin(angle);
			p.z = 0.0f;
			cloud->push_back(p);
		}
		return cloud;
	}


	// TransformStamped -> Eigen Matrix4f
	Eigen::Matrix4f transformToMatrix(const geometry_msgs::msg::TransformStamped & tf)
	{
		const auto & t = tf.transform.translation;
		const auto & q = tf.transform.rotation;
		Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
		Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
		mat.block<3,3>(0,0) = quat.toRotationMatrix();
		mat(0,3) = t.x;
		mat(1,3) = t.y;
		mat(2,3) = t.z;
		return mat;
	}

	// Eigen Matrix4f -> TransformStamped
	geometry_msgs::msg::TransformStamped matrixToTransform(const Eigen::Matrix4f & mat)
	{
		Eigen::Quaternionf quat(mat.block<3,3>(0,0));
		quat.normalize();
		geometry_msgs::msg::TransformStamped tf;
		tf.transform.translation.x = mat(0,3);
		tf.transform.translation.y = mat(1,3);
		tf.transform.translation.z = mat(2,3);
		tf.transform.rotation.x = quat.x();
		tf.transform.rotation.y = quat.y();
		tf.transform.rotation.z = quat.z();
		tf.transform.rotation.w = quat.w();
		return tf;
	}

	sensor_msgs::msg::PointCloud2 cloudToROSMsg(const CloudT & cloud, const std::string & frame, const rclcpp::Time & stamp)
	{
		sensor_msgs::msg::PointCloud2 msg;
		msg.header.stamp    = stamp;
		msg.header.frame_id = frame;
		msg.height          = 1;
		msg.width           = cloud.size();
		msg.is_dense        = false;
		msg.is_bigendian    = false;

		// Describe the XYZ fields
		sensor_msgs::msg::PointField fx, fy, fz;
		fx.name = "x"; fx.offset = 0;  fx.datatype = sensor_msgs::msg::PointField::FLOAT32; fx.count = 1;
		fy.name = "y"; fy.offset = 4;  fy.datatype = sensor_msgs::msg::PointField::FLOAT32; fy.count = 1;
		fz.name = "z"; fz.offset = 8;  fz.datatype = sensor_msgs::msg::PointField::FLOAT32; fz.count = 1;
		msg.fields = {fx, fy, fz};

		msg.point_step = 12;  // 3 * float32
		msg.row_step   = msg.point_step * msg.width;
		msg.data.resize(msg.row_step);

		for (size_t i = 0; i < cloud.size(); ++i) {
			uint8_t * ptr = msg.data.data() + i * msg.point_step;
			memcpy(ptr + 0, &cloud[i].x, 4);
			memcpy(ptr + 4, &cloud[i].y, 4);
			memcpy(ptr + 8, &cloud[i].z, 4);
		}
		return msg;
	}

	void publishCloud(const CloudT::Ptr & cloud,
                  const std::string & frame,
                  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub)
	{
		auto msg = cloudToROSMsg(*cloud, frame, this->get_clock()->now());
		pub->publish(msg);
	}

	void addDistanceTraveled()
	{
		if (T_odom_base_last_ == Eigen::Matrix4f::Identity()) {
			return;
		}
		Eigen::Vector3f dt = T_odom_base_.block<3,1>(0,3)
							- T_odom_base_last_.block<3,1>(0,3);
		distance_travelled_ += dt.norm();
		distance_travelled_last_icp_ += dt.norm();
	}
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
	auto node = std::make_shared<Localization>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();

    return 0;
}