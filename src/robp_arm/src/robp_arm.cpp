// ROS
#include <rclcpp/rclcpp.hpp>
#include <robp_interfaces/msg/arm_control.hpp>
#include <robp_interfaces/msg/arm_feedback.hpp>
#include <std_msgs/msg/empty.hpp>

// STL
#include <chrono>
#include <memory>
#include <string>

// C library headers
#include <stdio.h>
#include <string.h>

// Linux headers
#include <errno.h>    // Error integer and strerror() function
#include <fcntl.h>    // Contains file controls like O_RDWR
#include <termios.h>  // Contains POSIX terminal control definitions
#include <unistd.h>   // write(), read(), close()

// using namespace std::chrono_literals;

// /* This example creates a subclass of Node and uses a fancy C++11 lambda
//  * function to shorten the callback syntax, at the expense of making the
//  * code somewhat more difficult to understand at first glance. */

// class Arm : public rclcpp::Node
// {
//  public:
// 	Arm() : Node("arm")
// 	{
// 		int serial_port = open("/dev/ttyUSB0", O_RDWR);

// 		// Check for errors
// 		if (0 > serial_port) {
// 			printf("Error %i from open: %s\n", errno, strerror(errno));
// 		}

// 		pub_ = this->create_publisher<robp_interfaces::msg::ArmFeedback>("arm/feedback", 10);

// 		auto move_callback = [this](robp_interfaces::msg::ArmControl::UniquePtr msg) -> void {
// 			last_msg_ = *msg;
// 		};
// 		sub_ = this->create_subscription<robp_interfaces::msg::ArmControl>("arm/control", 10,
// 		                                                                   move_callback);

// 		auto timer_callback = [this]() -> void {
// 			auto message = std_msgs::msg::String();
// 			message.data = "Hello, world! " + std::to_string(this->count_++);
// 			RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
// 			this->publisher_->publish(message);
// 		};
// 		timer_ = this->create_wall_timer(500ms, timer_callback);
// 	}

//  private:
// 	void feedback()
// 	{
// 		std::uint8_t action = 0b010;

// 		if (reset_) {
// 			last_msg_ = {};
// 			cur_msg_  = {};
// 			reset_    = false;
// 			action |= 0b100;
// 		} else if (...) {
// 		}

// 		// Write to serial port
// 		write(serial_port, &action, sizeof(action));

// 		std::string const start_seq = "BEGIN FEEDBACK";
// 		std::string const end_seq   = "END FEEDBACK";
// 		std::size_t       pos       = 0;
// 		while (pos < start_seq.size()) {
// 			char c;
// 			int e = read(serial_port, &c, sizeof(c));  // Note you should be checking the result

// 			if (-1 == e) {
// 				printf("Error %i from read: %s\n", errno, strerror(errno));
// 			} else if (0 == e) {
// 				printf("Read nothing!\n");
// 			}

// 			pos = start_seq[pos] == c ? pos + 1 : 0;
// 		}

// 		robp_interfaces::msg::ArmFeedback msg;
// 		int e = read(serial_port, msg.position.data(), sizeof(msg.position));

// 		if (-1 == e) {
// 			printf("Error %i from read: %s\n", errno, strerror(errno));
// 		} else if (0 == e) {
// 			printf("Read nothing!\n");
// 		}

// 		msg.header.frame_id = "arm_link";
// 		msg.header.stamp    = this->now();

// 		pub_->publish(msg);

// 		while (pos < 99) {
// 			read(fd, buffer + pos, 1);  // Note you should be checking the result
// 			if (buffer[pos] == '\n') break;
// 			pos++;
// 		}
// 	}

// 	void setupTTY()
// 	{
// 		// Create new termios struct, we call it 'tty' for convention
// 		struct termios tty;

// 		// Read in existing settings, and handle any error
// 		if (tcgetattr(serial_port, &tty) != 0) {
// 			printf("Error %i from tcgetattr: %s\n", errno, strerror(errno));
// 			return 1;
// 		}

// 		tty.c_cflag &= ~PARENB;   // Clear parity bit, disabling parity (most common)
// 		tty.c_cflag &= ~CSTOPB;   // Clear stop field, only one stop bit used in communication
// 		                          // (most common)
// 		tty.c_cflag &= ~CSIZE;    // Clear all bits that set the data size
// 		tty.c_cflag |= CS8;       // 8 bits per byte (most common)
// 		tty.c_cflag &= ~CRTSCTS;  // Disable RTS/CTS hardware flow control (most common)
// 		tty.c_cflag |= CREAD | CLOCAL;  // Turn on READ & ignore ctrl lines (CLOCAL = 1)

// 		tty.c_lflag &= ~ICANON;
// 		tty.c_lflag &= ~ECHO;    // Disable echo
// 		tty.c_lflag &= ~ECHOE;   // Disable erasure
// 		tty.c_lflag &= ~ECHONL;  // Disable new-line echo
// 		tty.c_lflag &= ~ISIG;    // Disable interpretation of INTR, QUIT and SUSP
// 		tty.c_iflag &= ~(IXON | IXOFF | IXANY);  // Turn off s/w flow ctrl
// 		tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR |
// 		                 ICRNL);  // Disable any special handling of received bytes

// 		tty.c_oflag &=
// 		    ~OPOST;  // Prevent special interpretation of output bytes (e.g. newline chars)
// 		tty.c_oflag &= ~ONLCR;  // Prevent conversion of newline to carriage return/line feed
// 		// tty.c_oflag &= ~OXTABS; // Prevent conversion of tabs to spaces (NOT PRESENT ON
// 		// LINUX) tty.c_oflag &= ~ONOEOT; // Prevent removal of C-d chars (0x004) in output
// 		// (NOT PRESENT ON LINUX)

// 		tty.c_cc[VTIME] = 10;  // Wait for up to 1s (10 deciseconds), returning as soon as any
// 		                       // data is received.
// 		tty.c_cc[VMIN] = 0;

// 		// Set in/out baud rate to be 9600
// 		cfsetispeed(&tty, B9600);
// 		cfsetospeed(&tty, B9600);

// 		// Save tty settings, also checking for error
// 		if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
// 			printf("Error %i from tcsetattr: %s\n", errno, strerror(errno));
// 			return 1;
// 		}
// 	}

//  private:
// 	rclcpp::Publisher<robp_interfaces::msg::ArmFeedback>::SharedPtr   pub_;
// 	rclcpp::Subscription<robp_interfaces::msg::ArmControl>::SharedPtr sub_;
// 	rclcpp::TimerBase::SharedPtr                                      timer_;

// 	robp_interfaces::msg::ArmControl  last_msg_{};
// 	robp_interfaces::msg::ArmFeedback cur_msg_{};
// };

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	// rclcpp::spin(std::make_shared<MinimalPublisher>());
	rclcpp::shutdown();
	return 0;
}