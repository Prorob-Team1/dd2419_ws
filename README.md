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