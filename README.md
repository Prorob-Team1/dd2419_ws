## Milestone 0

For the joystick (turn it on first!), you can run the following inside the pixi shell:

    ros2 run joy joy_node

Alternatively, use the pixi task we created in pixi.toml:

    pixi run joy

To remote control run

    pixi run phidgets
    pixi run rc_control
    pixi run odometry

You can also publish a static transform from map to odom:
    
    pixi run ros2 run tf2_ros static_transform_publisher --frame-id map --child-frame-id odom

pixi run arm
pixi run arm_camera
pixi run lidar
pixi run realsense
pixi run phidgets
pixi run ros2 run joy joy_node
pixi run phidgets
pixi run rc_control
pixi run odometry


Running the rosbag:

    pixi run ros2 bag play --read-ahead-queue-size 100 -l -r 1.0 --clock 100 --start-paused ~/dd2419_ws/bags/rosbag2_2026_02_04-15_24_09/

Static transform for the lidar:

    pixi run ros2 run tf2_ros static_transform_publisher --frame-id base_link --child-frame-id lidar_link


## Milestone 1
    pixi run ros2 launch robp_launch basics_launch.yaml
    pixi run ros2 launch robp_launch static_transform_launch.yaml
    pixi run ros2 run novigation random_dispatcher
    pixi run ros2 run detection detection
    pixi run ros2 run robp_arm arm_move_action_server

## Milestone 2
    pixi run rviz2 -d rviz_config.rviz 
    pixi run ros2 launch robp_launch basics_launch.yaml
    pixi run ros2 launch robp_launch navigation_launch.yaml
    pixi run ros2 run napping mapping
    pixi run ros2 run brian brain
    pixi run ros2 run arm_grasping arm_grasping
    pixi run ros2 run detection pointcloud_filter
    


    pixi run ros2 run brian dummy_server
    pixi run ros2 service call /Start_Grasping std_srvs/srv/Trigger

## Milestone 3
    pixi run rviz2 -d rviz_config.rviz 
    pixi run MS3
    pixi run brian

For debugging, comment out the specific part you want to debug in the launch file (MS3_launch.yaml)