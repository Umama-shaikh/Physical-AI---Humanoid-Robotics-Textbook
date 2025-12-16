#!/usr/bin/env python3

"""
ROS 2 rclpy Example

This example demonstrates the basic structure of a ROS 2 node using rclpy.
It shows how to create a node, declare parameters, and handle basic operations.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


class RclpyExampleNode(Node):
    """
    An example node demonstrating various rclpy features.
    """

    def __init__(self):
        super().__init__('rclpy_example_node')

        # Declare parameters with default values
        self.declare_parameter('example_param', 'default_value')
        self.declare_parameter('count_threshold', 10)

        # Get parameter values
        self.example_param = self.get_parameter('example_param').value
        self.count_threshold = self.get_parameter('count_threshold').value

        # Initialize variables
        self.counter = 0

        # Create a publisher with custom QoS settings
        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.publisher_ = self.create_publisher(String, 'rclpy_example_topic', qos_profile)

        # Create a timer that calls the callback every 1 second
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Log initialization
        self.get_logger().info(
            f'Rclpy example node initialized with param: {self.example_param}, '
            f'threshold: {self.count_threshold}'
        )

    def timer_callback(self):
        """Timer callback that publishes messages and demonstrates rclpy features."""
        # Create and publish a message
        msg = String()
        msg.data = f'Hello from rclpy example: {self.counter}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')

        # Increment counter
        self.counter += 1

        # Demonstrate parameter change handling
        if self.counter % 5 == 0:
            # Example of getting parameter at runtime
            current_param = self.get_parameter('example_param').value
            self.get_logger().info(f'Current parameter value: {current_param}')

        # Demonstrate threshold behavior
        if self.counter >= self.count_threshold:
            self.get_logger().info(f'Counter reached threshold of {self.count_threshold}')
            # Reset counter
            self.counter = 0


def main(args=None):
    """
    Main function that initializes the node and runs the example.
    """
    rclpy.init(args=args)

    # Create the example node
    rclpy_example_node = RclpyExampleNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(rclpy_example_node)
    except KeyboardInterrupt:
        rclpy_example_node.get_logger().info('Example node stopped by user')
    finally:
        # Clean up
        rclpy_example_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()