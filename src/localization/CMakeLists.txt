cmake_minimum_required(VERSION 3.8)
project(localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

#set(PCL_DIR /home/robot/dd2419_ws/.pixi/envs/default/share/pcl-1.15) # why on earth was this needed
find_package(PCL REQUIRED COMPONENTS common registration)

add_executable(localization src/localization.cpp)
target_include_directories(localization PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_compile_features(localization PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  localization
  "geometry_msgs"
  "rclcpp"
  "tf2_ros"
  "sensor_msgs"
)
target_link_libraries(localization
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)

install(TARGETS localization
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
