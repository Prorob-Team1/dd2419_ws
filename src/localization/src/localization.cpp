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


using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class Localization : public rclcpp::Node
{
public:
    Localization() : Node("localization"), is_initialized_(false),
                     T_map_odom_(Eigen::Matrix4f::Identity())
    {
        tf_buffer_      = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/lidar/scan", 10,
            std::bind(&Localization::scanCallback, this, std::placeholders::_1));
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
			"/phidgets/imu/data_raw", 1, 
			std::bind(&Localization::imuCallback, this, std::placeholders::_1));
		
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

        RCLCPP_INFO(this->get_logger(), "Localization node started. Waiting for initial pose on /initial_pose...");
    }

private:
    struct Keyframe {
        Eigen::Matrix4f T_map_lidar;    // lidar pose in map when keyframe was taken
        CloudT::Ptr     cloud_in_map;   // points already transformed to map frame
        bool            is_anchor = false;
    };

    // ---- ROS handles --------------------------------------------------------
    std::unique_ptr<tf2_ros::Buffer>               tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>    tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ref_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_guess_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_aligned_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
	rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscription_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    rclcpp::TimerBase::SharedPtr init_retry_timer_;

    // ---- State --------------------------------------------------------------
    bool            is_initialized_;
    Eigen::Matrix4f T_map_lidar_0_;   // true lidar pose in map at t=0 (from initial pose)
    Eigen::Matrix4f T_map_odom_;      // map->odom correction we publish
    Eigen::Matrix4f T_base_lidar_;    // static base_link->lidar_link
	Eigen::Matrix4f T_odom_base_;
	Eigen::Matrix4f T_odom_base_last_ = Eigen::Matrix4f::Identity();



    geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pending_initial_pose_;

	bool has_reference_cloud_ = false;
	Eigen::Matrix4f T_base_lidar_ = Eigen::Matrix4f::Identity(); 

	int reference_scan_count_ = 0;
	const int kReferenceScanTarget = 5;
    
    const float icpFitnessThreshold = 0.1f;
    const float maxAngularVelForGoodScan = 0.3f;
    const float icpInterpolationAlpha = 0.05f;
    const float icpCorrectionDistanceFromStart = 1.5f;


	double ang_vel_{0};



    void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        if (is_initialized_) return;
        pending_initial_pose_ = msg;
        tryInitialize();
    }

    void tryInitialize()
    {
        const auto & pos = pending_initial_pose_->pose.pose.position;
        const auto & q   = pending_initial_pose_->pose.pose.orientation;
        Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
        Eigen::Matrix4f T_map_base = Eigen::Matrix4f::Identity();
        T_map_base.block<3,3>(0,0) = quat.toRotationMatrix();
        T_map_base(0,3) = pos.x;
        T_map_base(1,3) = pos.y;
        T_map_base(2,3) = pos.z;

        try {
            auto tf_base_lidar = tf_buffer_->lookupTransform(
                "base_link", "lidar_link", tf2::TimePointZero);
            T_base_lidar_ = transformToMatrix(tf_base_lidar);
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN(this->get_logger(),
                        "Waiting for base_link->lidar_link: %s. Retrying...", e.what());
            scheduleRetry();
            return;
        }

        T_map_lidar_0_ = T_map_base * T_base_lidar_;
        T_map_odom_    = T_map_base;

        is_initialized_ = true;
        if (init_retry_timer_) {
            init_retry_timer_->cancel();
            init_retry_timer_ = nullptr;
        }

        RCLCPP_INFO(this->get_logger(), "Initialization complete.");
    }

    void scheduleRetry()
    {
        if (init_retry_timer_) return;
        init_retry_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            [this]() {
                tryInitialize();
            });
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
		ang_vel_ = msg->angular_velocity.z;
	}

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!is_initialized_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Not initialized yet, skipping scan.");
            return;
        }

        CloudT::Ptr scan_cloud = laserScanToCloud(msg);
        if (scan_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty scan, skipping.");
            return;
        }

        // --- Phase 1: accumulate anchor keyframe at the known start pose -----
        if (anchor_scan_count_ < kAnchorScanTarget) {
            accumulateAnchor(scan_cloud, T_map_lidar_0_);
            anchor_scan_count_++;

            if (anchor_scan_count_ == kAnchorScanTarget) {
                RCLCPP_INFO(this->get_logger(),
                            "Anchor keyframe ready with %zu points.",
                            keyframes_[0].cloud_in_map->size());
            }
            return;
        }

        // --- Get odometry-based initial guess for current lidar pose ---------
        try {
            auto tf = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
            T_odom_base_ = transformToMatrix(tf);
			addDistanceTraveled();
			T_odom_base_last_ = T_odom_base_;
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN(this->get_logger(), "Could not get odom->base_link: %s", e.what());
            return;
        }


		// T_map_lidar_1_odom = T_map_odom * T_odom_base * T_base_lidar
		Eigen::Matrix4f T_map_lidar_1_guess = T_map_odom_ * T_odom_base * T_base_lidar_;

		// if the robot is far away from the reference pose, then skip ICP
		float initial_distance = (T_map_lidar_1_guess.block<3,1>(0,3) - T_map_lidar_0_.block<3,1>(0,3)).norm();
		if (initial_distance > icpCorrectionDistanceFromStart) return;  // tune this threshold for your environment
			
		if (std::abs(ang_vel_) > maxAngularVelForGoodScan) return;

		CloudT::Ptr cloud_at_guess(new CloudT());
		pcl::transformPointCloud(*scan_cloud, *cloud_at_guess, T_map_lidar_1_guess);
		cloud_at_guess = filterCloud(cloud_at_guess);

		Eigen::Matrix4f T_map_lidar_icp;
		if (distance_travelled_last_icp_ > distanceToRunICP && std::abs(ang_vel_) < maxAngularVelForGoodScan) {

			pcl::IterativeClosestPoint<PointT, PointT> icp;
			icp.setInputTarget(submap);
			icp.setInputSource(scan_cloud);
			icp.setMaximumIterations(50);
			icp.setTransformationEpsilon(1e-6);
			icp.setMaxCorrespondenceDistance(0.4);   // tune for your environment

			CloudT aligned;
			icp.align(aligned, T_map_lidar_guess);

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

		Eigen::Matrix4f T_map_odom_icp = T_map_base_1_icp * T_odom_base.inverse();
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

    // -------------------------------------------------------------------------
    // Utility functions
    // -------------------------------------------------------------------------
    CloudT::Ptr laserScanToCloud(const sensor_msgs::msg::LaserScan::SharedPtr msg) const
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

    CloudT::Ptr filterCloud(const CloudT::Ptr & input) const
    {
        CloudT::Ptr filtered(new CloudT());
        pcl::RadiusOutlierRemoval<PointT> ror;
        ror.setInputCloud(input);
        ror.setRadiusSearch(0.20);
        ror.setMinNeighborsInRadius(2);
        ror.filter(*filtered);
        return filtered;
    }

    // Interpolate between two SE(3) transforms.
    // alpha=0 → A, alpha=1 → B.
    Eigen::Matrix4f interpolateTransform(const Eigen::Matrix4f & A,
                                         const Eigen::Matrix4f & B,
                                         float alpha) const
    {
        Eigen::Vector3f t = (1.f - alpha) * A.block<3,1>(0,3)
                          +        alpha  * B.block<3,1>(0,3);

        Eigen::Quaternionf qa(A.block<3,3>(0,0));
        Eigen::Quaternionf qb(B.block<3,3>(0,0));
        Eigen::Quaternionf q = qa.slerp(alpha, qb);

        Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
        result.block<3,3>(0,0) = q.toRotationMatrix();
        result.block<3,1>(0,3) = t;
        return result;
    }

    Eigen::Matrix4f transformToMatrix(const geometry_msgs::msg::TransformStamped & tf) const
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

    geometry_msgs::msg::TransformStamped matrixToTransform(const Eigen::Matrix4f & mat) const
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

    sensor_msgs::msg::PointCloud2 cloudToROSMsg(const CloudT & cloud,
                                                 const std::string & frame,
                                                 const rclcpp::Time & stamp) const
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp    = stamp;
        msg.header.frame_id = frame;
        msg.height          = 1;
        msg.width           = cloud.size();
        msg.is_dense        = false;
        msg.is_bigendian    = false;

        sensor_msgs::msg::PointField fx, fy, fz;
        fx.name = "x"; fx.offset = 0;  fx.datatype = sensor_msgs::msg::PointField::FLOAT32; fx.count = 1;
        fy.name = "y"; fy.offset = 4;  fy.datatype = sensor_msgs::msg::PointField::FLOAT32; fy.count = 1;
        fz.name = "z"; fz.offset = 8;  fz.datatype = sensor_msgs::msg::PointField::FLOAT32; fz.count = 1;
        msg.fields = {fx, fy, fz};

        msg.point_step = 12;
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
		// if we havent seen a lst post return
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
    rclcpp::spin(std::make_shared<Localization>());
    rclcpp::shutdown();
    return 0;
}