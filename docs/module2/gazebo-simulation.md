---
sidebar_position: 2
---

# Gazebo Physics Simulation

## Overview

This chapter introduces Gazebo, a powerful physics simulation environment for robotics. You'll learn how to create realistic simulation environments, configure physics properties, and integrate your ROS 2 systems with Gazebo for testing and development. Gazebo provides a "digital twin" of your robot and its environment, allowing you to test algorithms safely before deployment on real hardware.

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure Gazebo for ROS 2 integration
- Create and customize simulation environments with models and objects
- Configure physics properties and sensors for realistic simulation
- Launch simulation environments and interact with them through ROS 2
- Debug and optimize simulation performance

## Table of Contents

1. [Introduction to Gazebo](#introduction-to-gazebo)
2. [Installing and Setting Up Gazebo](#installing-and-setting-up-gazebo)
3. [Gazebo World Structure](#gazebo-world-structure)
4. [Creating Simulation Environments](#creating-simulation-environments)
5. [Physics Configuration](#physics-configuration)
6. [Sensor Integration](#sensor-integration)
7. [ROS 2 Integration](#ros-2-integration)
8. [Simulation Best Practices](#simulation-best-practices)
9. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides high-fidelity physics simulation, realistic rendering, and convenient programmatic interfaces.

### Key Features of Gazebo

- **Physics Engine**: Uses ODE (Open Dynamics Engine) for accurate physics simulation
- **Rendering**: Realistic 3D rendering using OGRE
- **Sensors**: Support for various robot sensors (cameras, LiDAR, IMU, etc.)
- **Plugins**: Extensible architecture through plugins
- **ROS Integration**: Seamless integration with ROS and ROS 2

### Why Use Simulation?

Simulation is crucial for robotics development because it:
- Allows safe testing of algorithms without risk to hardware
- Enables rapid iteration and debugging
- Provides reproducible testing conditions
- Supports development of multiple robots simultaneously
- Reduces development costs

## Installing and Setting Up Gazebo

### Installation

For ROS 2 Humble Hawksbill, install Gazebo Garden:

```bash
# Install Gazebo Garden
sudo apt-get update
sudo apt-get install gazebo-garden

# Install ROS 2 Gazebo packages
sudo apt-get install ros-humble-gazebo-ros-pkgs
sudo apt-get install ros-humble-gazebo-plugins
sudo apt-get install ros-humble-gazebo-dev
```

### Basic Gazebo Commands

```bash
# Launch Gazebo GUI
gazebo

# Launch Gazebo with a specific world
gazebo /path/to/world.world

# Launch Gazebo in headless mode (no GUI)
gz sim -s -r /path/to/world.sdf
```

## Gazebo World Structure

### World File Format

![Simulation Architecture](../../assets/diagrams/simulation_architecture.svg)

Gazebo uses SDF (Simulation Description Format) files to describe simulation worlds. SDF is an XML-based format that can describe:

- World properties (gravity, magnetic field, etc.)
- Models and their placement
- Lighting and visual effects
- Physics engine configuration
- Plugins

### Basic World File

Here's a minimal Gazebo world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- World properties -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.0 -1.0</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- A simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.083</iyy>
            <iyz>0</iyz>
            <izz>0.083</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.1 0.1 0.4 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
            <specular>0.4 0.4 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Creating Simulation Environments

### Environment Design Principles

When creating simulation environments:

1. **Start Simple**: Begin with basic environments and gradually add complexity
2. **Realistic Properties**: Use materials and physics properties that match reality
3. **Proper Scaling**: Ensure objects are appropriately sized relative to your robot
4. **Lighting Considerations**: Configure lighting for sensor simulation
5. **Performance**: Balance realism with simulation performance

### Common Environment Elements

#### Rooms and Buildings

```xml
<!-- Room with walls -->
<model name="room">
  <static>true</static>
  <!-- Floor -->
  <link name="floor">
    <collision name="floor_collision">
      <geometry><box><size>10 10 0.1</size></box></geometry>
    </collision>
    <visual name="floor_visual">
      <geometry><box><size>10 10 0.1</size></box></geometry>
    </visual>
  </link>

  <!-- Walls -->
  <link name="wall_north">
    <pose>0 5 2.5 0 0 0</pose>
    <collision name="wall_north_collision">
      <geometry><box><size>10 0.2 5</size></box></geometry>
    </collision>
    <visual name="wall_north_visual">
      <geometry><box><size>10 0.2 5</size></box></geometry>
    </visual>
  </link>
  <!-- Add other walls similarly -->
</model>
```

#### Obstacles and Furniture

```xml
<!-- Table -->
<model name="table">
  <pose>2 0 0 0 0 0</pose>
  <link name="top">
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>1.0</ixx> <ixy>0</ixy> <ixz>0</ixz>
        <iyy>1.0</iyy> <iyz>0</iyz> <izz>1.0</izz>
      </inertia>
    </inertial>
    <collision name="top_collision">
      <geometry><box><size>1.5 0.8 0.05</size></box></geometry>
    </collision>
    <visual name="top_visual">
      <geometry><box><size>1.5 0.8 0.05</size></box></geometry>
    </visual>
  </link>
  <link name="leg1">
    <pose>0.6 0.3 -0.4 0 0 0</pose>
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.1</ixx> <ixy>0</ixy> <ixz>0</ixz>
        <iyy>0.1</iyy> <iyz>0</iyz> <izz>0.1</izz>
      </inertia>
    </inertial>
    <collision name="leg1_collision">
      <geometry><box><size>0.05 0.05 0.8</size></box></geometry>
    </collision>
    <visual name="leg1_visual">
      <geometry><box><size>0.05 0.05 0.8</size></box></geometry>
    </visual>
  </link>
  <!-- Add other legs -->
</model>
```

## Physics Configuration

### Physics Engine Parameters

The physics engine can be configured with various parameters:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Time step -->
  <real_time_factor>1</real_time_factor>  <!-- Real-time simulation -->
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Material Properties

Configure surface properties for realistic interactions:

```xml
<!-- In a link's collision element -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>  <!-- Coefficient of friction -->
      <mu2>1.0</mu2>
      <fdir1>0 0 0</fdir1>
      <slip1>0</slip1>
      <slip2>0</slip2>
    </ode>
  </friction>
  <bounce>
    <restitution_coefficient>0.1</restitution_coefficient>
    <threshold>100000</threshold>
  </bounce>
  <contact>
    <ode>
      <soft_cfm>0</soft_cfm>
      <soft_erp>0.2</soft_erp>
      <kp>1e+13</kp>
      <kd>1</kd>
      <max_vel>100</max_vel>
      <min_depth>0</min_depth>
    </ode>
  </contact>
</surface>
```

## Sensor Integration

### Camera Sensor

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="camera">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
    <topic_name>image_raw</topic_name>
  </plugin>
</sensor>
```

### LiDAR Sensor

```xml
<sensor name="lidar" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <topic_name>scan</topic_name>
    <frame_name>lidar_frame</frame_name>
  </plugin>
</sensor>
```

### IMU Sensor

```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <topic_name>imu</topic_name>
    <frame_name>imu_frame</frame_name>
  </plugin>
</sensor>
```

## ROS 2 Integration

### Launching Gazebo with ROS 2

Create a launch file to start Gazebo with your robot:

```python
# launch/gazebo.launch.py
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_description = get_package_share_directory('my_robot_description')

    # Start Gazebo server
    start_gazebo_server_cmd = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-s',
            os.path.join(pkg_robot_description, 'worlds', 'my_world.sdf')
        ],
        output='screen'
    )

    # Start Gazebo client (GUI)
    start_gazebo_client_cmd = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': open(
                os.path.join(pkg_robot_description, 'urdf', 'robot.urdf')
            ).read()}
        ]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot'
        ],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add commands to launch description
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_entity)

    return ld
```

### Controlling Robots in Simulation

To control your robot in simulation, you'll typically publish to the same topics as you would for a real robot:

```python
# example_robot_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math


class SimulationController(Node):
    def __init__(self):
        super().__init__('simulation_controller')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for laser scan (for obstacle avoidance)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Initialize variables
        self.scan_data = None
        self.forward_speed = 0.5
        self.rotation_speed = 0.5

    def scan_callback(self, msg):
        self.scan_data = msg

    def control_loop(self):
        if self.scan_data is None:
            return

        # Simple obstacle avoidance
        min_distance = min(self.scan_data.ranges)

        cmd = Twist()

        if min_distance < 0.5:  # Too close to obstacle
            # Stop and rotate
            cmd.linear.x = 0.0
            cmd.angular.z = self.rotation_speed
        else:
            # Move forward
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    controller = SimulationController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Simulation Best Practices

### Performance Optimization

1. **Reduce Update Rates**: Lower sensor update rates when possible
2. **Simplify Models**: Use simpler collision geometries for performance
3. **Optimize Physics**: Adjust time steps and solver parameters
4. **Limit Visuals**: Reduce rendering quality during headless simulation

### Accuracy Considerations

1. **Match Real Robot**: Use identical URDF models for simulation and real robot
2. **Tune Parameters**: Calibrate sensor noise and physics properties
3. **Validate Results**: Compare simulation and real-world behavior
4. **Document Differences**: Note any simulation limitations

### Debugging Tips

1. **Visualize TF**: Use RViz to visualize transforms
2. **Monitor Topics**: Check that all expected topics are published
3. **Use Gazebo GUI**: Visualize the simulation to spot issues
4. **Log Messages**: Add logging to track robot behavior

## Practical Example: Simple Navigation in Gazebo

Let's create a complete example of a robot navigating in a simple environment:

1. **Create the world file** (`worlds/simple_room.sdf`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.0 -1.0</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal></plane></geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <model name="room_walls">
      <static>true</static>
      <link name="wall_north">
        <pose>0 5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>10 0.2 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>10 0.2 3</size></box></geometry>
        </visual>
      </link>
      <link name="wall_south">
        <pose>0 -5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>10 0.2 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>10 0.2 3</size></box></geometry>
        </visual>
      </link>
      <link name="wall_east">
        <pose>5 0 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>0.2 10 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 10 3</size></box></geometry>
        </visual>
      </link>
      <link name="wall_west">
        <pose>-5 0 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>0.2 10 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 10 3</size></box></geometry>
        </visual>
      </link>
    </model>

    <!-- A simple obstacle -->
    <model name="obstacle">
      <pose>-2 2 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.416</ixx> <ixy>0</ixy> <ixz>0</ixz>
            <iyy>0.416</iyy> <iyz>0</iyz> <izz>0.833</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>1 1 1</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>1 1 1</size></box></geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>1.0 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

2. **Launch the simulation**:

```bash
# Terminal 1: Launch Gazebo with the world
gz sim -r worlds/simple_room.sdf

# Terminal 2: Launch your robot controller
ros2 run my_package simulation_controller
```

## Summary and Next Steps

In this chapter, you learned:
- How to install and configure Gazebo for ROS 2 integration
- How to create and customize simulation environments
- How to configure physics properties for realistic simulation
- How to integrate sensors and control systems with Gazebo
- Best practices for simulation development and debugging

### Key Takeaways

- Simulation is a crucial tool for robotics development and testing
- Gazebo provides a realistic physics environment for robot testing
- Proper configuration of physics and sensor properties is essential
- ROS 2 integration allows seamless transition between simulation and real robots
- Performance optimization is important for complex simulations

### Next Steps

In the next chapter, you'll learn about simulated sensors and how to configure them properly for realistic perception in your simulation environments.

## Exercises

1. Create a Gazebo world with multiple rooms and obstacles
2. Configure a robot with camera and LiDAR sensors in simulation
3. Implement a simple navigation algorithm that works in both simulation and reality
4. Compare sensor data from simulation with real sensor data
5. Optimize your simulation for better performance

## References

- Gazebo Documentation: http://gazebosim.org/
- Gazebo ROS Packages: http://wiki.ros.org/gazebo_ros_pkgs
- SDF Specification: http://sdformat.org/
- ODE Physics Engine: http://ode.org/