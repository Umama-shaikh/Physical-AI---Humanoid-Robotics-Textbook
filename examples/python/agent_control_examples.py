#!/usr/bin/env python3

"""
Python Agent Control Examples for ROS 2

This file contains various examples of Python agents that control robots
using ROS 2. These examples demonstrate different control patterns and
strategies for robot operation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

import math
import numpy as np
from enum import Enum
from collections import deque
import time


class RobotState(Enum):
    """Robot operational states"""
    IDLE = 1
    MOVING = 2
    TURNING = 3
    AVOIDING = 4
    EMERGENCY_STOP = 5
    NAVIGATING = 6
    MANIPULATING = 7


class SimpleNavigationAgent(Node):
    """
    A simple navigation agent that moves forward until it detects obstacles,
    then turns to avoid them.
    """

    def __init__(self):
        super().__init__('simple_navigation_agent')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Parameters
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('safety_distance', 0.5)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # State
        self.scan_data = None
        self.odom_data = None
        self.current_state = RobotState.IDLE
        self.obstacle_detected = False

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Simple Navigation Agent initialized')

    def scan_callback(self, msg):
        """Callback for laser scan data"""
        self.scan_data = msg
        self.check_for_obstacles()

    def odom_callback(self, msg):
        """Callback for odometry data"""
        self.odom_data = msg

    def check_for_obstacles(self):
        """Check if there are obstacles in the path"""
        if self.scan_data is None:
            return

        # Get ranges in front of the robot (forward 90 degrees)
        front_ranges = self.scan_data.ranges[len(self.scan_data.ranges)//2 - 45 : len(self.scan_data.ranges)//2 + 45]

        # Filter out invalid ranges
        valid_ranges = [r for r in front_ranges if r > 0.1 and r < 10.0]

        if valid_ranges:
            min_range = min(valid_ranges)
            self.obstacle_detected = min_range < self.safety_distance
        else:
            self.obstacle_detected = False

    def control_loop(self):
        """Main control loop"""
        cmd = Twist()

        if self.obstacle_detected:
            # Turn to avoid obstacle
            self.current_state = RobotState.AVOIDING
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        else:
            # Move forward
            self.current_state = RobotState.MOVING
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

        self.get_logger().debug(f'State: {self.current_state}, Obstacle: {self.obstacle_detected}')


class PatrolAgent(Node):
    """
    A patrol agent that follows a predefined path, avoiding obstacles.
    """

    def __init__(self):
        super().__init__('patrol_agent')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Parameters
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('safety_distance', 0.6)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Patrol waypoints (x, y coordinates)
        self.waypoints = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0)
        ]

        self.current_waypoint_index = 0
        self.target_tolerance = 0.3  # meters

        # State
        self.scan_data = None
        self.odom_data = None
        self.current_state = RobotState.NAVIGATING
        self.obstacle_detected = False
        self.current_position = Point()

        # Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Patrol Agent initialized')

    def scan_callback(self, msg):
        self.scan_data = msg
        self.check_for_obstacles()

    def odom_callback(self, msg):
        self.odom_data = msg
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y

    def check_for_obstacles(self):
        if self.scan_data is None:
            return

        # Check for obstacles in front
        front_ranges = self.scan_data.ranges[len(self.scan_data.ranges)//2 - 30 : len(self.scan_data.ranges)//2 + 30]
        valid_ranges = [r for r in front_ranges if r > 0.1 and r < 10.0]

        if valid_ranges:
            min_range = min(valid_ranges)
            self.obstacle_detected = min_range < self.safety_distance
        else:
            self.obstacle_detected = False

    def get_current_waypoint(self):
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        else:
            # Return to first waypoint after completing patrol
            return self.waypoints[0]

    def control_loop(self):
        cmd = Twist()

        if self.obstacle_detected:
            # Emergency stop or turn to avoid
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
            self.current_state = RobotState.AVOIDING
        else:
            # Navigate to current waypoint
            target_x, target_y = self.get_current_waypoint()

            # Calculate distance to target
            dx = target_x - self.current_position.x
            dy = target_y - self.current_position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < self.target_tolerance:
                # Reached current waypoint, move to next
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_index}, moving to next')

            # Calculate heading to target
            target_angle = math.atan2(dy, dx)
            current_angle = self.get_yaw_from_quaternion(self.odom_data.pose.pose.orientation)

            # Calculate angle difference
            angle_diff = target_angle - current_angle
            # Normalize angle difference to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # If robot is roughly facing the target, move forward
            if abs(angle_diff) < 0.2:  # 0.2 radians = ~11 degrees
                cmd.linear.x = self.linear_speed
                cmd.angular.z = 0.0
            else:
                # Turn towards target
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed

        self.cmd_vel_pub.publish(cmd)

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)


class MultiSensorFusionAgent(Node):
    """
    An agent that fuses data from multiple sensors for navigation.
    """

    def __init__(self):
        super().__init__('multi_sensor_fusion_agent')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.viz_pub = self.create_publisher(Marker, '/obstacle_marker', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Parameters
        self.declare_parameter('linear_speed', 0.4)
        self.declare_parameter('angular_speed', 0.6)
        self.declare_parameter('safety_distance', 0.7)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Data storage
        self.scan_data = None
        self.imu_data = None
        self.odom_data = None

        # State
        self.current_state = RobotState.IDLE
        self.obstacle_positions = []  # Store detected obstacle positions

        # Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Multi-Sensor Fusion Agent initialized')

    def scan_callback(self, msg):
        self.scan_data = msg
        self.process_lidar_data()

    def imu_callback(self, msg):
        self.imu_data = msg

    def odom_callback(self, msg):
        self.odom_data = msg

    def process_lidar_data(self):
        """Process LiDAR data to detect obstacles and their positions"""
        if self.scan_data is None or self.odom_data is None:
            return

        ranges = np.array(self.scan_data.ranges)
        angles = np.linspace(
            self.scan_data.angle_min,
            self.scan_data.angle_max,
            len(ranges)
        )

        # Convert ranges and angles to Cartesian coordinates relative to robot
        valid_indices = np.where((ranges > 0.1) & (ranges < 10.0) & (ranges < self.safety_distance))[0]

        obstacle_positions = []
        for idx in valid_indices:
            angle = angles[idx]
            range_val = ranges[idx]

            # Convert to robot frame coordinates
            x_local = range_val * math.cos(angle)
            y_local = range_val * math.sin(angle)

            # Transform to global frame using robot pose
            robot_x = self.odom_data.pose.pose.position.x
            robot_y = self.odom_data.pose.pose.position.y
            robot_yaw = self.get_yaw_from_quaternion(self.odom_data.pose.pose.orientation)

            x_global = robot_x + x_local * math.cos(robot_yaw) - y_local * math.sin(robot_yaw)
            y_global = robot_y + x_local * math.sin(robot_yaw) + y_local * math.cos(robot_yaw)

            obstacle_positions.append((x_global, y_global))

        self.obstacle_positions = obstacle_positions

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        cmd = Twist()

        if self.scan_data is None:
            return

        # Check for obstacles in front
        front_ranges = self.scan_data.ranges[len(self.scan_data.ranges)//2 - 30 : len(self.scan_data.ranges)//2 + 30]
        valid_ranges = [r for r in front_ranges if r > 0.1 and r < 10.0]

        if valid_ranges and min(valid_ranges) < self.safety_distance:
            # Obstacle detected, use IMU data for stabilization
            cmd.linear.x = 0.0

            if self.imu_data:
                # Use IMU to help with turning (counteract drift)
                angular_vel_z = self.imu_data.angular_velocity.z
                cmd.angular.z = self.angular_speed - angular_vel_z * 0.1
            else:
                cmd.angular.z = self.angular_speed

            self.current_state = RobotState.AVOIDING
        else:
            # Clear path, move forward with IMU stabilization
            cmd.linear.x = self.linear_speed

            if self.imu_data:
                # Use IMU to maintain straight line
                angular_vel_z = self.imu_data.angular_velocity.z
                cmd.angular.z = -angular_vel_z * 0.5  # Small correction
            else:
                cmd.angular.z = 0.0

            self.current_state = RobotState.MOVING

        self.cmd_vel_pub.publish(cmd)
        self.publish_obstacle_markers()

    def publish_obstacle_markers(self):
        """Publish obstacle positions as visualization markers"""
        if not self.obstacle_positions:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for x, y in self.obstacle_positions:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.1  # Slightly above ground
            marker.points.append(point)

        self.viz_pub.publish(marker)


class BehaviorTreeAgent(Node):
    """
    An agent that uses a simple behavior tree for decision making.
    """

    def __init__(self):
        super().__init__('behavior_tree_agent')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twest, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Parameters
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('battery_threshold', 20.0)  # percent

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.battery_threshold = self.get_parameter('battery_threshold').value

        # State
        self.scan_data = None
        self.odom_data = None
        self.battery_level = 100.0  # Simulated battery level
        self.charging_station_pos = (5.0, 5.0)  # Simulated charging station position

        # Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Behavior Tree Agent initialized')

    def scan_callback(self, msg):
        self.scan_data = msg

    def odom_callback(self, msg):
        self.odom_data = msg

    def check_battery_level(self):
        """Simulate battery level checking (in real system, this would come from robot)"""
        # Simulate battery drain
        self.battery_level -= 0.01  # Small drain per call
        return self.battery_level

    def check_obstacles(self):
        """Check for obstacles in front of the robot"""
        if self.scan_data is None:
            return False

        front_ranges = self.scan_data.ranges[len(self.scan_data.ranges)//2 - 45 : len(self.scan_data.ranges)//2 + 45]
        valid_ranges = [r for r in front_ranges if r > 0.1 and r < 10.0]

        if valid_ranges and min(valid_ranges) < self.safety_distance:
            return True
        return False

    def get_distance_to_charging_station(self):
        """Calculate distance to charging station"""
        if self.odom_data is None:
            return float('inf')

        robot_x = self.odom_data.pose.pose.position.x
        robot_y = self.odom_data.pose.pose.position.y

        dx = self.charging_station_pos[0] - robot_x
        dy = self.charging_station_pos[1] - robot_y

        return math.sqrt(dx*dx + dy*dy)

    def move_to_charging_station(self):
        """Simple navigation to charging station"""
        if self.odom_data is None:
            return Twist()

        cmd = Twist()

        robot_x = self.odom_data.pose.pose.position.x
        robot_y = self.odom_data.pose.pose.position.y
        robot_yaw = self.get_yaw_from_quaternion(self.odom_data.pose.pose.orientation)

        # Calculate direction to charging station
        dx = self.charging_station_pos[0] - robot_x
        dy = self.charging_station_pos[1] - robot_y
        target_angle = math.atan2(dy, dx)

        angle_diff = target_angle - robot_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # If roughly aligned, move forward
        if abs(angle_diff) < 0.2:
            cmd.linear.x = self.linear_speed * 0.7  # Slower for precision
            cmd.angular.z = 0.0
        else:
            # Turn towards charging station
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed

        return cmd

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        """Execute behavior tree logic"""
        cmd = Twist()

        # Behavior tree: sequence of conditions and actions
        battery_level = self.check_battery_level()
        has_obstacles = self.check_obstacles()
        distance_to_charging = self.get_distance_to_charging_station()

        # Root behavior: if battery is low, go to charging station
        if battery_level < self.battery_threshold:
            self.get_logger().info(f'Battery low ({battery_level:.1f}%), going to charging station')
            cmd = self.move_to_charging_station()
        # If there are obstacles, avoid them
        elif has_obstacles:
            self.get_logger().info('Obstacle detected, turning to avoid')
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        # If close to charging station and battery is low, stop (for charging)
        elif distance_to_charging < 0.5 and battery_level < 50.0:
            self.get_logger().info('At charging station')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        # Otherwise, continue normal operation (in this example, patrol behavior)
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    """
    Main function to demonstrate different agent types.
    Usage: ros2 run your_package agent_control_examples.py [agent_type]
    agent_type: simple, patrol, fusion, behavior
    """
    import sys

    rclpy.init(args=args)

    agent_type = 'simple'  # default
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].lower()

    if agent_type == 'patrol':
        agent = PatrolAgent()
    elif agent_type == 'fusion':
        agent = MultiSensorFusionAgent()
    elif agent_type == 'behavior':
        agent = BehaviorTreeAgent()
    else:  # default to simple
        agent = SimpleNavigationAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info(f'{agent_type.capitalize()} agent stopped by user')
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()