---
sidebar_position: 3
---

# Python Agents → ROS Controllers (rclpy)

## Overview

This chapter focuses on connecting Python agents to ROS controllers using rclpy, the Python client library for ROS 2. You'll learn how to create intelligent agents that can interact with robotic systems, bridging the gap between high-level Python logic and low-level robot control.

## Learning Objectives

By the end of this chapter, you will be able to:
- Use rclpy to create Python nodes that interact with ROS systems
- Implement control logic in Python that interfaces with robot controllers
- Design agent-based architectures for robot control
- Create custom message types for agent-robot communication
- Implement feedback control loops using ROS communication patterns

## Table of Contents

1. [Introduction to rclpy](#introduction-to-rclpy)
2. [Setting Up Python Agents](#setting-up-python-agents)
3. [Connecting to Robot Controllers](#connecting-to-robot-controllers)
4. [Agent-Based Control Patterns](#agent-based-control-patterns)
5. [Custom Messages and Services](#custom-messages-and-services)
6. [Feedback Control with ROS](#feedback-control-with-ros)
7. [Practical Examples](#practical-examples)
8. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing Python APIs for creating ROS nodes, publishers, subscribers, services, and actions. It allows Python developers to leverage their existing Python skills to create sophisticated robotic applications.

### Why Use Python for Agent Development?

Python is ideal for creating intelligent agents due to its:
- Rich ecosystem of machine learning and AI libraries
- Easy-to-read syntax that facilitates rapid prototyping
- Strong support for data processing and analysis
- Extensive scientific computing libraries
- Integration capabilities with various systems

### Core rclpy Concepts

The main components of rclpy include:

- **Node**: The basic execution entity
- **Publisher**: Sends messages to topics
- **Subscriber**: Receives messages from topics
- **Client**: Calls services
- **Service**: Provides service callbacks
- **Timer**: Executes callbacks at regular intervals
- **Parameter**: Configuration values

## Setting Up Python Agents

### Basic Node Structure

A typical Python agent node follows this structure:

```python
import rclpy
from rclpy.node import Node

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # Initialize parameters
        self.declare_parameter('agent_name', 'default_agent')
        self.agent_name = self.get_parameter('agent_name').value

        # Create publishers, subscribers, services, etc.
        self.setup_communication()

        # Initialize agent state
        self.agent_state = 'idle'

        # Log initialization
        self.get_logger().info(f'Agent {self.agent_name} initialized')

    def setup_communication(self):
        # Set up publishers, subscribers, etc.
        pass

def main(args=None):
    rclpy.init(args=args)

    agent_node = AgentNode()

    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        agent_node.get_logger().info('Agent interrupted by user')
    finally:
        agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Agent State Management

Agents often need to maintain internal state. Here's an example of state management:

```python
from enum import Enum

class AgentState(Enum):
    IDLE = 1
    PLANNING = 2
    EXECUTING = 3
    ERROR = 4

class StatefulAgent(Node):
    def __init__(self):
        super().__init__('stateful_agent')
        self.current_state = AgentState.IDLE
        self.previous_state = None

    def change_state(self, new_state):
        self.previous_state = self.current_state
        self.current_state = new_state
        self.get_logger().info(f'State changed from {self.previous_state} to {self.current_state}')
```

## Connecting to Robot Controllers

### Publisher-Controller Interface

The most common pattern is using publishers to send commands to robot controllers:

```python
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class MotionAgent(Node):
    def __init__(self):
        super().__init__('motion_agent')

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Publisher for joint positions
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        # Timer for periodic control updates
        self.control_timer = self.create_timer(
            0.1,  # 10 Hz
            self.control_loop
        )

    def control_loop(self):
        # Implement your control logic here
        if self.current_state == 'moving_to_goal':
            cmd = self.compute_velocity_command()
            self.cmd_vel_publisher.publish(cmd)
```

### Subscriber for Sensor Feedback

Subscribers allow agents to receive feedback from the robot:

```python
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class PerceptionAgent(Node):
    def __init__(self):
        super().__init__('perception_agent')

        # Subscribe to laser scan data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Subscribe to odometry data
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Initialize sensor data storage
        self.laser_data = None
        self.odom_data = None

    def scan_callback(self, msg):
        self.laser_data = msg
        self.process_scan_data()

    def odom_callback(self, msg):
        self.odom_data = msg
        self.update_position_estimate()

    def process_scan_data(self):
        # Process laser scan data
        if self.laser_data:
            # Example: detect obstacles
            min_distance = min(self.laser_data.ranges)
            if min_distance < 0.5:  # 0.5 meters
                self.get_logger().warn('Obstacle detected!')
```

### Service-Based Control

Services provide synchronous control when needed:

```python
from example_interfaces.srv import Trigger

class ControlAgent(Node):
    def __init__(self):
        super().__init__('control_agent')

        # Create client for emergency stop service
        self.emergency_stop_client = self.create_client(
            Trigger,
            '/emergency_stop'
        )

        # Wait for service to be available
        while not self.emergency_stop_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Emergency stop service not available, waiting...')

    def emergency_stop(self):
        request = Trigger.Request()
        future = self.emergency_stop_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
```

## Agent-Based Control Patterns

### Behavior Trees

Behavior trees are a popular pattern for organizing agent behavior:

```python
class BehaviorTreeAgent(Node):
    def __init__(self):
        super().__init__('behavior_tree_agent')
        self.current_behavior = None

    def run_behavior_tree(self):
        # Example behavior tree logic
        if self.check_battery_level() < 0.2:
            self.current_behavior = 'return_to_charger'
        elif self.has_pending_tasks():
            self.current_behavior = 'execute_task'
        else:
            self.current_behavior = 'idle'

        self.execute_current_behavior()

    def check_battery_level(self):
        # Implement battery level checking
        return 0.8  # Example value

    def has_pending_tasks(self):
        # Implement task checking
        return True  # Example value

    def execute_current_behavior(self):
        if self.current_behavior == 'return_to_charger':
            self.return_to_charger()
        elif self.current_behavior == 'execute_task':
            self.execute_task()
        elif self.current_behavior == 'idle':
            self.idle_behavior()
```

### Finite State Machines

FSMs are useful for agents with distinct operational modes:

```python
from enum import Enum

class RobotMode(Enum):
    IDLE = 1
    NAVIGATING = 2
    MANIPULATING = 3
    CHARGING = 4

class StateMachineAgent(Node):
    def __init__(self):
        super().__init__('state_machine_agent')
        self.current_mode = RobotMode.IDLE
        self.mode_timer = self.create_timer(0.1, self.state_machine_loop)

    def state_machine_loop(self):
        if self.current_mode == RobotMode.IDLE:
            if self.received_navigation_goal():
                self.transition_to_navigation()
        elif self.current_mode == RobotMode.NAVIGATING:
            if self.reached_goal():
                self.transition_to_idle()
            elif self.encountered_obstacle():
                self.transition_to_idle()
        # Add other state transitions...

    def transition_to_navigation(self):
        self.current_mode = RobotMode.NAVIGATING
        self.get_logger().info('Transitioned to NAVIGATING mode')

    def transition_to_idle(self):
        self.current_mode = RobotMode.IDLE
        self.get_logger().info('Transitioned to IDLE mode')
```

## Custom Messages and Services

### Creating Custom Messages

Custom messages allow agents to communicate complex information:

```python
# In a custom_msgs package, create msg/AgentCommand.msg:
# string agent_id
# string command_type
# float64[] parameters
# string[] string_params
```

### Using Custom Messages

```python
from custom_msgs.msg import AgentCommand

class CustomMessageAgent(Node):
    def __init__(self):
        super().__init__('custom_message_agent')

        self.command_publisher = self.create_publisher(
            AgentCommand,
            'agent_commands',
            10
        )

    def send_agent_command(self, agent_id, cmd_type, params=None):
        cmd = AgentCommand()
        cmd.agent_id = agent_id
        cmd.command_type = cmd_type
        cmd.parameters = params or []
        cmd.string_params = []

        self.command_publisher.publish(cmd)
```

## Feedback Control with ROS

### PID Controller Example

Here's an example of implementing a PID controller in an agent:

```python
class PIDAgent(Node):
    def __init__(self):
        super().__init__('pid_agent')

        # PID parameters
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.previous_time = self.get_clock().now()

        # Publisher for control commands
        self.control_publisher = self.create_publisher(
            Float64MultiArray,
            '/control_commands',
            10
        )

    def compute_pid_control(self, setpoint, current_value):
        # Calculate error
        error = setpoint - current_value

        # Calculate time delta
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9
        self.previous_time = current_time

        if dt <= 0:
            return 0.0

        # Calculate integral and derivative
        self.error_integral += error * dt
        derivative = (error - self.previous_error) / dt

        # Calculate control output
        output = (self.kp * error +
                 self.ki * self.error_integral +
                 self.kd * derivative)

        # Store current error for next iteration
        self.previous_error = error

        return output
```

## Practical Examples

### Simple Navigation Agent

Here's a complete example of a navigation agent:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math

class NavigationAgent(Node):
    def __init__(self):
        super().__init__('navigation_agent')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.goal_subscriber = self.create_subscription(
            PoseStamped, '/goal', self.goal_callback, 10)

        # Parameters
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 0.5)
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        # State
        self.laser_data = None
        self.current_goal = None
        self.robot_pose = None

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        self.laser_data = msg

    def goal_callback(self, msg):
        self.current_goal = msg.pose
        self.get_logger().info(f'New goal received: {msg.pose.position.x}, {msg.pose.position.y}')

    def control_loop(self):
        if self.current_goal is None or self.laser_data is None:
            return

        # Check for obstacles
        if self.detect_obstacle():
            self.stop_robot()
            return

        # Calculate control commands to reach goal
        cmd = self.calculate_navigation_command()
        self.cmd_vel_publisher.publish(cmd)

    def detect_obstacle(self):
        if self.laser_data is None:
            return False

        # Check if there are obstacles within 0.5m
        min_distance = min(self.laser_data.ranges)
        return min_distance < 0.5

    def calculate_navigation_command(self):
        cmd = Twist()

        # Simple proportional controller for navigation
        goal_x = self.current_goal.position.x
        goal_y = self.current_goal.position.y

        # Calculate distance and angle to goal (simplified)
        distance = math.sqrt(goal_x**2 + goal_y**2)
        angle_to_goal = math.atan2(goal_y, goal_x)

        # Simple control logic
        if distance > 0.1:  # If not close to goal
            cmd.linear.x = min(self.linear_speed, distance * 0.5)
            cmd.angular.z = angle_to_goal * 0.5
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    agent = NavigationAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Navigation agent stopped by user')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary and Next Steps

In this chapter, you learned how to:
- Create Python agents using rclpy
- Connect agents to robot controllers through various communication patterns
- Implement agent-based control patterns like behavior trees and state machines
- Design feedback control systems that interact with ROS
- Create practical navigation and control examples

### Key Takeaways

- Python agents bridge high-level logic with low-level robot control
- rclpy provides the interface between Python and ROS 2
- Proper state management is crucial for robust agent behavior
- Feedback control systems ensure reliable robot operation
- Custom messages enable complex agent-robot interactions

### Next Steps

In the next chapter, you'll learn about URDF (Unified Robot Description Format) and how to model humanoid robots, which will complement the control techniques learned here.

## Exercises

1. Create an agent that implements a simple patrol behavior
2. Design a custom message for task assignment between agents
3. Implement a PID controller for joint position control
4. Create a state machine agent that handles multiple robot modes
5. Build a simple path planning agent using potential fields

## References

- rclpy Documentation: https://docs.ros.org/en/rolling/p/rclpy/
- ROS 2 Python Tutorials: https://docs.ros.org/en/rolling/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Node.html
- Behavior Trees in Robotics: https://arxiv.org/abs/1709.00084