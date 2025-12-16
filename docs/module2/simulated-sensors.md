---
sidebar_position: 3
---

# Simulated Sensors (LiDAR, Depth, IMU)

## Overview

This chapter focuses on simulating various types of robot sensors in Gazebo, including LiDAR, depth cameras, and IMUs. You'll learn how to configure these sensors with realistic parameters, understand their characteristics, and integrate them with ROS 2 for perception tasks. Proper sensor simulation is crucial for developing and testing perception algorithms before deployment on real robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Configure and integrate LiDAR sensors in Gazebo simulation
- Set up depth cameras and understand their properties
- Implement IMU sensors with realistic noise characteristics
- Understand the differences between simulated and real sensor data
- Optimize sensor configurations for performance and accuracy
- Validate sensor data for perception algorithms

## Table of Contents

1. [Introduction to Simulated Sensors](#introduction-to-simulated-sensors)
2. [LiDAR Sensors in Gazebo](#lidar-sensors-in-gazebo)
3. [Depth Camera Sensors](#depth-camera-sensors)
4. [IMU Sensors](#imu-sensors)
5. [Sensor Noise and Realism](#sensor-noise-and-realism)
6. [ROS 2 Sensor Integration](#ros-2-sensor-integration)
7. [Sensor Validation and Testing](#sensor-validation-and-testing)
8. [Performance Optimization](#performance-optimization)
9. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to Simulated Sensors

### Why Simulate Sensors?

Sensor simulation is essential for robotics development because it:
- Allows testing of perception algorithms without expensive hardware
- Provides ground truth data for algorithm validation
- Enables controlled testing of edge cases
- Supports development of multiple robots simultaneously
- Reduces development time and costs

### Sensor Simulation Challenges

- **Realism**: Simulated sensors should behave similarly to real sensors
- **Performance**: Complex sensor simulation can impact simulation speed
- **Calibration**: Sensor parameters should match real hardware when possible
- **Noise**: Realistic noise models are crucial for robust algorithms

## LiDAR Sensors in Gazebo

### Understanding LiDAR Sensors

LiDAR (Light Detection and Ranging) sensors emit laser beams and measure the time it takes for the light to return after reflecting off objects. In simulation, Gazebo uses ray tracing to simulate this process.

### LiDAR Configuration in SDF

```xml
<sensor name="lidar" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>  <!-- Number of rays in horizontal scan -->
        <resolution>1</resolution>  <!-- Resolution of rays -->
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees in radians -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees in radians -->
      </horizontal>
      <vertical>
        <samples>1</samples>  <!-- For 2D LiDAR, set to 1 -->
        <resolution>1</resolution>
        <min_angle>0</min_angle>  <!-- For 2D LiDAR, set to 0 -->
        <max_angle>0</max_angle>   <!-- For 2D LiDAR, set to 0 -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>  <!-- Minimum detectable range -->
      <max>30.0</max>  <!-- Maximum detectable range -->
      <resolution>0.01</resolution>  <!-- Range resolution -->
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <topic_name>scan</topic_name>
    <frame_name>lidar_frame</frame_name>
    <min_intensity>0.0</min_intensity>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</sensor>
```

### 3D LiDAR Configuration

For 3D LiDAR sensors like Velodyne:

```xml
<sensor name="velodyne" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- 360 degrees -->
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>32</samples>  <!-- Number of laser beams (for VLP-16: 16, VLP-32: 32) -->
        <resolution>1</resolution>
        <min_angle>-0.436332</min_angle>  <!-- -25 degrees in radians -->
        <max_angle>0.244346</max_angle>   <!-- 14 degrees in radians -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu.so">
    <topic_name>points</topic_name>
    <frame_name>velodyne_frame</frame_name>
    <min_range>0.1</min_range>
    <max_range>100.0</max_range>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</sensor>
```

### LiDAR Processing in ROS 2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def lidar_callback(self, msg):
        # Process LiDAR data
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        # Calculate minimum distance (obstacle detection)
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Minimum distance: {min_distance:.2f}m')

        # Calculate distances in specific sectors
        sector_size = len(ranges) // 8  # Divide into 8 sectors
        for i in range(8):
            start_idx = i * sector_size
            end_idx = min((i + 1) * sector_size, len(ranges))
            sector_ranges = ranges[start_idx:end_idx]
            valid_sector = sector_ranges[np.isfinite(sector_ranges)]

            if len(valid_sector) > 0:
                avg_distance = np.mean(valid_sector)
                self.get_logger().info(f'Sector {i}: avg distance = {avg_distance:.2f}m')


def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        lidar_processor.get_logger().info('Lidar processor stopped by user')
    finally:
        lidar_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Depth Camera Sensors

### Understanding Depth Cameras

Depth cameras provide both color images and depth information for each pixel. They are essential for 3D perception, mapping, and navigation tasks.

### Depth Camera Configuration in SDF

```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_cam">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>  <!-- Near clipping plane -->
      <far>10.0</far>   <!-- Far clipping plane -->
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="depth_camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>depth_camera_frame</frame_name>
    <topic_name>image_raw</topic_name>
    <depth_topic_name>depth/image_raw</depth_topic_name>
    <point_cloud_topic_name>points</point_cloud_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
  </plugin>
</sensor>
```

### Depth Camera Processing in ROS 2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')
        self.bridge = CvBridge()

        # Subscribe to depth image
        self.depth_subscription = self.create_subscription(
            Image,
            'depth/image_raw',
            self.depth_callback,
            10
        )

        # Subscribe to color image
        self.image_subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )

        # Store camera info for 3D point calculation
        self.camera_info = None
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            10
        )

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Calculate depth statistics
            valid_depths = depth_image[np.isfinite(depth_image) & (depth_image > 0)]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)

                self.get_logger().info(
                    f'Depth stats - Avg: {avg_depth:.2f}m, '
                    f'Min: {min_depth:.2f}m, Max: {max_depth:.2f}m'
                )

            # Find obstacles in specific regions
            height, width = depth_image.shape
            center_region = depth_image[
                height//4:3*height//4,
                width//4:3*width//4
            ]

            center_depths = center_region[np.isfinite(center_region) & (center_region > 0)]
            if len(center_depths) > 0 and np.min(center_depths) < 1.0:  # Obstacle within 1m
                self.get_logger().warn('Obstacle detected in front!')

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Process the image (example: edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Display results (optional, for visualization)
            # cv2.imshow('Depth Camera', cv_image)
            # cv2.imshow('Edges', edges)
            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    processor = DepthCameraProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Depth camera processor stopped by user')
    finally:
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## IMU Sensors

### Understanding IMU Sensors

An IMU (Inertial Measurement Unit) typically contains accelerometers, gyroscopes, and sometimes magnetometers. It provides information about the robot's orientation, angular velocity, and linear acceleration.

### IMU Configuration in SDF

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <!-- Gyroscope noise parameters -->
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>  <!-- Standard deviation in rad/s -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </angular_velocity>

    <!-- Accelerometer noise parameters -->
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>  <!-- Standard deviation in m/s^2 -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.01</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.01</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.01</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <topic_name>imu</topic_name>
    <frame_name>imu_link</frame_name>
    <body_name>base_link</body_name>
    <update_rate>100</update_rate>
  </plugin>
</sensor>
```

### IMU Processing in ROS 2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import math


class ImuProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')
        self.subscription = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Store previous orientation for change detection
        self.prev_orientation = None
        self.orientation_change_threshold = 0.1  # radians

    def imu_callback(self, msg):
        # Extract orientation (using quaternion to Euler conversion)
        orientation = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # Log orientation
        self.get_logger().info(
            f'Orientation - Roll: {math.degrees(roll):.2f}°, '
            f'Pitch: {math.degrees(pitch):.2f}°, '
            f'Yaw: {math.degrees(yaw):.2f}°'
        )

        # Extract angular velocity
        angular_velocity = msg.angular_velocity
        self.get_logger().info(
            f'Angular Vel - X: {angular_velocity.x:.3f}, '
            f'Y: {angular_velocity.y:.3f}, '
            f'Z: {angular_velocity.z:.3f}'
        )

        # Extract linear acceleration
        linear_accel = msg.linear_acceleration
        self.get_logger().info(
            f'Linear Accel - X: {linear_accel.x:.3f}, '
            f'Y: {linear_accel.y:.3f}, '
            f'Z: {linear_accel.z:.3f}'
        )

        # Detect significant orientation changes
        if self.prev_orientation is not None:
            delta_roll = abs(roll - self.prev_orientation[0])
            delta_pitch = abs(pitch - self.prev_orientation[1])
            delta_yaw = abs(yaw - self.prev_orientation[2])

            if (delta_roll > self.orientation_change_threshold or
                delta_pitch > self.orientation_change_threshold or
                delta_yaw > self.orientation_change_threshold):
                self.get_logger().info('Significant orientation change detected!')

        self.prev_orientation = (roll, pitch, yaw)

    def quaternion_to_euler(self, x, y, z, w):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    imu_processor = ImuProcessor()

    try:
        rclpy.spin(imu_processor)
    except KeyboardInterrupt:
        imu_processor.get_logger().info('IMU processor stopped by user')
    finally:
        imu_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Sensor Noise and Realism

### Understanding Sensor Noise

Real sensors have inherent noise that affects their measurements. Simulating this noise is crucial for developing robust algorithms.

### Noise Models

#### Gaussian Noise
Most sensors exhibit Gaussian-distributed noise:

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.01</stddev>
</noise>
```

#### Bias and Drift
Sensors may have systematic errors:

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.01</stddev>
  <bias_mean>0.001</bias_mean>  <!-- Systematic offset -->
  <bias_stddev>0.0001</bias_stddev>  <!-- Bias instability -->
</noise>
```

### Adding Custom Noise in Python

```python
import numpy as np

def add_noise_to_sensor_data(data, noise_std, bias=0.0):
    """
    Add realistic noise to sensor data
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, data.shape)

    # Add bias
    biased_data = data + bias

    # Add noise
    noisy_data = biased_data + noise

    return noisy_data

# Example usage
real_distance = 2.5  # meters
noisy_distance = add_noise_to_sensor_data(
    np.array([real_distance]),
    noise_std=0.02,  # 2cm standard deviation
    bias=0.01        # 1cm bias
)[0]

print(f'Real: {real_distance:.2f}m, Noisy: {noisy_distance:.2f}m')
```

## ROS 2 Sensor Integration

### Sensor Message Types

ROS 2 defines standard message types for different sensors:

- **LaserScan**: For LiDAR and 2D range finders
- **Image**: For camera images
- **PointCloud2**: For 3D point cloud data
- **Imu**: For inertial measurement units
- **MagneticField**: For magnetometers
- **FluidPressure**: For pressure sensors
- **Illuminance**: For light sensors
- **Temperature**: For temperature sensors
- **RelativeHumidity**: For humidity sensors

### Multi-Sensor Fusion Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import Twist
import numpy as np
from collections import deque


class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Subscribers for different sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, 'image_raw', self.camera_callback, 10)

        # Publisher for fused commands
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Data storage with history
        self.lidar_history = deque(maxlen=5)
        self.imu_history = deque(maxlen=10)

        # Timer for fusion logic
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)

        # Robot state
        self.safety_distance = 0.5  # meters

    def lidar_callback(self, msg):
        # Store LiDAR data
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.lidar_history.append(min_distance)

    def imu_callback(self, msg):
        # Store IMU data
        self.imu_history.append({
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        })

    def camera_callback(self, msg):
        # Process camera data (simplified)
        pass

    def fusion_callback(self):
        # Implement sensor fusion logic
        cmd = Twist()

        # Check for obstacles using LiDAR
        if len(self.lidar_history) > 0:
            recent_distances = list(self.lidar_history)
            min_recent_distance = min(recent_distances)

            if min_recent_distance < self.safety_distance:
                # Emergency stop if obstacle too close
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().warn('Emergency stop: obstacle detected!')
            else:
                # Normal navigation behavior
                cmd.linear.x = 0.3  # Move forward slowly

                # Use IMU data for stabilization
                if len(self.imu_history) > 0:
                    latest_imu = self.imu_history[-1]
                    # Simple angular correction based on IMU
                    angular_vel_z = latest_imu['angular_velocity'].z
                    cmd.angular.z = -angular_vel_z * 0.5  # Counteract rotation

        # Publish fused command
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultiSensorFusion()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Multi-sensor fusion stopped by user')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Sensor Validation and Testing

### Comparing Simulated vs Real Data

![Sensor Integration](../../assets/diagrams/sensor_integration.svg)

To validate sensor simulation:

1. **Collect real sensor data** from your actual robot
2. **Configure simulation** with the same parameters
3. **Compare statistical properties** of the data
4. **Validate perception algorithms** work on both datasets

### Data Analysis Example

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_sensor_data(sim_data, real_data, sensor_type):
    """
    Compare simulated and real sensor data
    """
    # Statistical comparison
    sim_mean = np.mean(sim_data)
    real_mean = np.mean(real_data)

    sim_std = np.std(sim_data)
    real_std = np.std(real_data)

    print(f"{sensor_type} Comparison:")
    print(f"  Simulated - Mean: {sim_mean:.3f}, Std: {sim_std:.3f}")
    print(f"  Real      - Mean: {real_mean:.3f}, Std: {real_std:.3f}")
    print(f"  Difference - Mean: {abs(sim_mean - real_mean):.3f}")
    print(f"  Difference - Std: {abs(sim_std - real_std):.3f}")

    # Plot comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(sim_data, label='Simulated', alpha=0.7)
    plt.plot(real_data, label='Real', alpha=0.7)
    plt.title(f'{sensor_type} - Time Series')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(sim_data, bins=50, alpha=0.5, label='Simulated', density=True)
    plt.hist(real_data, bins=50, alpha=0.5, label='Real', density=True)
    plt.title(f'{sensor_type} - Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()
```

## Performance Optimization

### Sensor Performance Tips

1. **Reduce Update Rates**: Lower sensor update rates when high frequency isn't needed
2. **Simplify Sensor Models**: Use fewer rays for LiDAR when resolution isn't critical
3. **Optimize Image Resolution**: Use lower resolution images during development
4. **Selective Publishing**: Only publish sensor data when needed
5. **Use Efficient Data Structures**: Process sensor data efficiently

### Optimized Sensor Configuration

```xml
<!-- Optimized LiDAR for performance -->
<sensor name="lidar_optimized" type="ray">
  <update_rate>5</update_rate>  <!-- Lower update rate -->
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>  <!-- Half the resolution -->
        <resolution>1</resolution>
        <min_angle>-1.57</min_angle>  <!-- 90 degrees FOV instead of 180 -->
        <max_angle>1.57</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>  <!-- Reduced range -->
      <resolution>0.05</resolution>  <!-- Lower resolution -->
    </range>
  </ray>
</sensor>
```

## Practical Example: Obstacle Detection System

Here's a complete example combining multiple sensors for obstacle detection:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
import numpy as np
from enum import Enum


class RobotState(Enum):
    IDLE = 1
    MOVING_FORWARD = 2
    TURNING = 3
    EMERGENCY_STOP = 4


class ObstacleDetectionSystem(Node):
    def __init__(self):
        super().__init__('obstacle_detection_system')

        # Subscriptions
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.viz_pub = self.create_publisher(Marker, 'obstacle_marker', 10)

        # Parameters
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)

        self.safety_distance = self.get_parameter('safety_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        # State
        self.lidar_data = None
        self.imu_data = None
        self.current_state = RobotState.IDLE
        self.obstacle_positions = []  # For visualization

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def lidar_callback(self, msg):
        self.lidar_data = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def control_loop(self):
        if self.lidar_data is None:
            return

        # Process LiDAR data to detect obstacles
        ranges = np.array(self.lidar_data.ranges)
        angles = np.linspace(
            self.lidar_data.angle_min,
            self.lidar_data.angle_max,
            len(ranges)
        )

        # Find obstacles within safety distance
        obstacle_indices = np.where((ranges < self.safety_distance) & (ranges > 0.1))[0]

        if len(obstacle_indices) > 0:
            # Calculate obstacle positions in robot frame
            obstacle_angles = angles[obstacle_indices]
            obstacle_distances = ranges[obstacle_indices]

            obstacle_x = obstacle_distances * np.cos(obstacle_angles)
            obstacle_y = obstacle_distances * np.sin(obstacle_angles)

            self.obstacle_positions = list(zip(obstacle_x, obstacle_y))

            # Change state to turning
            self.current_state = RobotState.TURNING
        else:
            # No obstacles, move forward
            self.current_state = RobotState.MOVING_FORWARD

        # Execute state-based behavior
        cmd = Twist()

        if self.current_state == RobotState.MOVING_FORWARD:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif self.current_state == RobotState.TURNING:
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        elif self.current_state == RobotState.EMERGENCY_STOP:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Publish command
        self.cmd_pub.publish(cmd)

        # Publish visualization
        self.publish_obstacle_markers()

    def publish_obstacle_markers(self):
        if not self.obstacle_positions:
            return

        marker = Marker()
        marker.header.frame_id = "base_link"
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
            point.z = 0.0
            marker.points.append(point)

        self.viz_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = ObstacleDetectionSystem()

    try:
        rclpy.spin(obstacle_detector)
    except KeyboardInterrupt:
        obstacle_detector.get_logger().info('Obstacle detection system stopped by user')
    finally:
        obstacle_detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary and Next Steps

In this chapter, you learned:
- How to configure and use LiDAR sensors in Gazebo
- How to set up depth cameras and IMUs with realistic parameters
- How to process sensor data in ROS 2 nodes
- How to implement sensor fusion for perception tasks
- How to optimize sensor configurations for performance
- How to validate sensor simulation against real hardware

### Key Takeaways

- Sensor simulation is crucial for developing robust perception algorithms
- Realistic noise models are essential for algorithm robustness
- Proper sensor configuration affects both realism and performance
- Multi-sensor fusion can provide more reliable perception
- Validation against real hardware is important for simulation accuracy

### Next Steps

In the next chapter, you'll learn about Unity for Human-Robot Interaction, exploring how to create intuitive interfaces for robot operation and monitoring.

## Exercises

1. Create a Gazebo world with multiple sensor types and test their integration
2. Implement a SLAM algorithm using simulated sensor data
3. Compare the performance of different LiDAR configurations
4. Create a sensor fusion node that combines data from multiple sensors
5. Develop a calibration procedure for simulated sensors to match real hardware

## References

- Gazebo Sensor Documentation: http://gazebosim.org/tutorials?tut=ros_gzplugins#Sensor-plugins
- ROS 2 Sensor Message Types: http://docs.ros.org/en/rolling/p/sensor_msgs/
- Sensor Noise Modeling: https://en.wikipedia.org/wiki/Sensor