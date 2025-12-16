#!/usr/bin/env python3

"""
ROS 2 Publisher-Subscriber Example

This example demonstrates the basic publisher-subscriber communication pattern in ROS 2.
It includes both a publisher node that sends messages and a subscriber node that receives them.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """
    A publisher node that sends messages to a topic.
    """

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        """Callback function that executes at regular intervals."""
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


class MinimalSubscriber(Node):
    """
    A subscriber node that receives messages from a topic.
    """

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        """Callback function that executes when a message is received."""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main_publisher(args=None):
    """
    Main function for the publisher node.
    """
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        minimal_publisher.get_logger().info('Publisher stopped by user')
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()


def main_subscriber(args=None):
    """
    Main function for the subscriber node.
    """
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        minimal_subscriber.get_logger().info('Subscriber stopped by user')
    finally:
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys

    # Check command line arguments to determine which node to run
    if len(sys.argv) > 1:
        if sys.argv[1] == 'publisher':
            main_publisher()
        elif sys.argv[1] == 'subscriber':
            main_subscriber()
        else:
            print("Usage: ros2_publisher_subscriber.py [publisher|subscriber]")
    else:
        print("Please specify 'publisher' or 'subscriber' as an argument")
        print("Example: python ros2_publisher_subscriber.py publisher")
        print("Example: python ros2_publisher_subscriber.py subscriber")