---
sidebar_position: 4
---

# Nav2 for Humanoids: Navigation in Complex Environments

## Overview

This chapter explores the implementation of Navigation2 (Nav2) for humanoid robots, addressing the unique challenges of bipedal locomotion, balance management, and terrain navigation. You'll learn how to configure Nav2 for humanoid-specific requirements, implement footstep planning, and integrate balance control with navigation systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Configure Nav2 for humanoid robot navigation
- Implement footstep planning algorithms for bipedal locomotion
- Integrate balance control with navigation systems
- Handle terrain challenges specific to humanoid robots
- Optimize navigation performance for real-time humanoid applications

## Table of Contents

1. [Introduction to Humanoid Navigation](#introduction-to-humanoid-navigation)
2. [Nav2 Architecture for Humanoids](#nav2-architecture-for-humanoids)
3. [Configuration for Bipedal Locomotion](#configuration-for-bipedal-locomotion)
4. [Path Planning for Humanoids](#path-planning-for-humanoids)
5. [Trajectory Execution and Footstep Planning](#trajectory-execution-and-footstep-planning)
6. [Balance and Stability Considerations](#balance-and-stability-considerations)
7. [Perception Integration](#perception-integration)
8. [Recovery Behaviors](#recovery-behaviors)
9. [Testing and Validation](#testing-and-validation)
10. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to Humanoid Navigation

### Unique Challenges of Humanoid Navigation

Humanoid robot navigation presents distinct challenges compared to wheeled or tracked robots:

1. **Balance Management**: Maintaining center of mass (CoM) stability during movement
2. **Footstep Planning**: Calculating safe and stable foot placements
3. **Terrain Adaptation**: Handling uneven surfaces, stairs, and obstacles
4. **Dynamic Stability**: Managing balance during transitions and turns
5. **Computational Constraints**: Real-time balance and navigation computation

### Humanoid-Specific Navigation Requirements

Unlike traditional mobile robots, humanoid robots must consider:
- **Zero Moment Point (ZMP)**: Ensuring stable foot placement
- **Capture Point**: Managing dynamic balance during walking
- **Footstep Constraints**: Limited step size and placement options
- **Balance Recovery**: Quick responses to perturbations
- **Multi-terrain Adaptation**: Different walking patterns for various surfaces

## Nav2 Architecture for Humanoids

### Modified Navigation Stack

The traditional Nav2 stack requires modifications for humanoid robots:

```
Navigation Stack for Humanoids
├── Global Planner (Humanoid-aware)
│   ├── Terrain analysis for bipedal locomotion
│   ├── Footstep planning integration
│   └── Balance constraint consideration
├── Local Planner (Humanoid-specific)
│   ├── Dynamic balance checking
│   ├── Footstep execution
│   └── Real-time balance recovery
├── Controller (Bipedal Controller)
│   ├── Footstep tracking
│   ├── Balance maintenance
│   └── Walking pattern generation
├── Perception (Humanoid-aware)
│   ├── Ground plane detection
│   ├── Obstacle height analysis
│   └── Terrain classification
└── Recovery Behaviors (Balance-focused)
    ├── Balance recovery
    ├── Fall prevention
    └── Safe stopping
```

### Humanoid-Specific Plugins

Nav2 for humanoids requires specialized plugins:

1. **Global Planner**: HumanoidGlobalPlanner with ZMP constraints
2. **Local Planner**: HumanoidLocalPlanner with balance checking
3. **Controller**: FootstepController for bipedal locomotion
4. **Costmap**: 3D costmap with terrain analysis
5. **Behavior Tree**: Humanoid-specific navigation behavior tree

## Configuration for Bipedal Locomotion

### Nav2 Configuration File

```yaml
# nav2_params_humanoid.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: False
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: True
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    default_bt_xml_filename: "humanoid_navigator_bt.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.4
      vx_max: 0.4
      vx_min: -0.2
      vy_max: 0.1
      wz_max: 0.4
      sim_period: 0.05
      goal_dist_threshold: 0.25
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory: True
      publish_cost_grid_pc: False
      transform_tolerance: 0.1
      speed_regulator_enabled: False
      max_speed_reference: 1.0
      speed_lim_vx_max: 0.8
      speed_lim_vx_min: -0.4
      speed_lim_vy_max: 0.4
      noise_factor: 0.8
      reference_trajectory_duration: 1.0
      path_dist_gain: 16.0
      goal_dist_gain: 24.0
      occ_cost_gain: 1.5
      feature_reg_gain: 0.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Humanoid-specific radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3  # Humanoid-specific radius
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.6

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      # Humanoid-aware global planner
      plugin: "humanoid_nav2_plugins::HumanoidGlobalPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      planner_frequency: 1.0
      max_iterations: 10000
      max_planning_time: 5.0
      smoothness_cost_weight: 0.1
      obstacle_cost_weight: 1.0
      dynamic_obstacle_inflation_dist: 0.6
      costmap_weight: 1.0
      humanoid_step_height: 0.15
      humanoid_step_length: 0.4
      humanoid_step_width: 0.25
      balance_margin: 0.1

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      wait_time: 1
```

### Humanoid-Specific Parameters

The configuration includes humanoid-specific parameters:
- **Robot Radius**: Adjusted for humanoid dimensions (0.3m)
- **Step Constraints**: Maximum step height (0.15m), length (0.4m), width (0.25m)
- **Balance Margin**: Safety margin for balance constraints (0.1m)
- **Velocity Limits**: Reduced for stability (0.4 m/s max forward speed)
- **3D Costmap**: Voxel layer for terrain analysis

## Path Planning for Humanoids

### Humanoid-Aware Global Planner

```python
# humanoid_global_planner.py
import numpy as np
from nav2_core.global_planner import GlobalPlanner
from nav2_costmap_2d.costmap_2d_ros import Costmap2DROS
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer
import math

class HumanoidGlobalPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.step_height_limit = 0.15  # 15cm max step height
        self.step_length_limit = 0.4   # 40cm max step length
        self.balance_margin = 0.1      # 10cm balance safety margin
        self.foot_size = 0.25          # 25cm foot width

    def configure(self, tf_buffer, costmap_ros, planner_node):
        """Configure the planner with costmap and TF buffer"""
        self.tf_buffer = tf_buffer
        self.costmap = costmap_ros
        self.planner_node = planner_node
        self.logger = planner_node.get_logger()

    def cleanup(self):
        """Clean up resources"""
        pass

    def set_costmap(self, costmap_ros):
        """Set the costmap for planning"""
        self.costmap = costmap_ros

    def create_plan(self, start, goal):
        """Create a navigation plan considering humanoid constraints"""
        path = Path()
        path.header.stamp = self.planner_node.get_clock().now().to_msg()
        path.header.frame_id = self.costmap.get_global_frame_id()

        # Check if start and goal are valid
        if not self.is_valid_pose(start) or not self.is_valid_pose(goal):
            self.logger.error("Invalid start or goal pose for humanoid navigation")
            return path

        # Check terrain between start and goal
        if not self.is_traversable_terrain(start, goal):
            self.logger.error("Path contains non-traversable terrain for humanoid")
            return path

        # Plan path using modified A* that considers humanoid constraints
        planned_path = self.humanoid_astar(start, goal)

        if planned_path:
            path.poses = planned_path
            self.logger.info(f"Successfully planned path with {len(planned_path)} waypoints")
        else:
            self.logger.error("Failed to find valid path for humanoid navigation")

        return path

    def is_valid_pose(self, pose):
        """Check if a pose is valid for humanoid navigation"""
        costmap = self.costmap.get_costmap()
        mx, my = self.pose_to_map_coordinates(pose)

        # Check if pose is in costmap bounds
        if mx < 0 or mx >= costmap.size_x or my < 0 or my >= costmap.size_y:
            return False

        # Check cost at pose location
        cost = costmap.get_cost(mx, my)
        if cost >= 253:  # lethal obstacle
            return False

        return True

    def is_traversable_terrain(self, start, goal):
        """Check if terrain between start and goal is traversable for humanoid"""
        # Check for stairs, steep slopes, or gaps that exceed humanoid capabilities
        # This is a simplified check - in practice, you'd analyze elevation data
        start_x, start_y = start.pose.position.x, start.pose.position.y
        goal_x, goal_y = goal.pose.position.x, goal.pose.position.y

        # Calculate path segments and check each for humanoid constraints
        distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        steps = max(10, int(distance / 0.1))  # Check every 10cm

        for i in range(steps + 1):
            t = i / steps
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)

            # Check elevation changes and obstacles at each point
            if not self.is_humanoid_safe_point(x, y):
                return False

        return True

    def is_humanoid_safe_point(self, x, y):
        """Check if a point is safe for humanoid navigation"""
        costmap = self.costmap.get_costmap()
        mx, my = self.world_to_map(x, y)

        if mx < 0 or mx >= costmap.size_x or my < 0 or my >= costmap.size_y:
            return False

        cost = costmap.get_cost(mx, my)
        if cost >= 200:  # High cost area
            return False

        # Additional checks for humanoid-specific constraints
        # (e.g., step height, surface type, etc.)
        return True

    def humanoid_astar(self, start, goal):
        """A* algorithm modified for humanoid constraints"""
        # Implementation of A* with humanoid-specific constraints
        # This would include checking step constraints, balance requirements, etc.
        # For brevity, returning a straight-line path as an example
        path = []

        # Simple straight-line path generation
        start_pos = [start.pose.position.x, start.pose.position.y]
        goal_pos = [goal.pose.position.x, goal.pose.position.y]

        distance = math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)
        num_points = max(10, int(distance / 0.2))  # Points every 20cm

        for i in range(num_points + 1):
            t = i / num_points
            x = start_pos[0] + t * (goal_pos[0] - start_pos[0])
            y = start_pos[1] + t * (goal_pos[1] - start_pos[1])

            pose = PoseStamped()
            pose.header.frame_id = self.costmap.get_global_frame_id()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path.append(pose)

        return path

    def world_to_map(self, wx, wy):
        """Convert world coordinates to map coordinates"""
        costmap = self.costmap.get_costmap()
        mx = int((wx - costmap.origin_x) / costmap.resolution)
        my = int((wy - costmap.origin_y) / costmap.resolution)
        return mx, my

    def pose_to_map_coordinates(self, pose):
        """Convert pose to map coordinates"""
        return self.world_to_map(pose.pose.position.x, pose.pose.position.y)
```

### Local Planner for Humanoid Navigation

```python
# humanoid_local_planner.py
import numpy as np
from nav2_core.controller import Controller
from nav2_util import LifecycleNode
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer
import math

class HumanoidLocalPlanner(Controller):
    def __init__(self):
        super().__init__()
        self.current_path = None
        self.current_waypoint_idx = 0
        self.balance_margin = 0.1
        self.step_size_limit = 0.4
        self.max_linear_vel = 0.3  # Reduced for stability
        self.max_angular_vel = 0.2

    def configure(self, tf_buffer, costmap_ros, local_planer_node):
        """Configure the local planner"""
        self.tf_buffer = tf_buffer
        self.costmap = costmap_ros
        self.local_planner_node = local_planer_node
        self.logger = local_planer_node.get_logger()

    def cleanup(self):
        """Clean up resources"""
        pass

    def setPlan(self, path):
        """Set the global plan for local execution"""
        self.current_path = path
        self.current_waypoint_idx = 0

    def setSpeedLimit(self, speed_limit, percentage):
        """Set speed limits for humanoid navigation"""
        if percentage:
            self.max_linear_vel *= speed_limit
            self.max_angular_vel *= speed_limit
        else:
            self.max_linear_vel = min(speed_limit, 0.3)  # Cap at safe humanoid speed

    def computeVelocityCommands(self, pose, velocity):
        """Compute velocity commands considering balance constraints"""
        cmd_vel = Twist()

        if not self.current_path or len(self.current_path.poses) == 0:
            return cmd_vel, Time()

        # Get next waypoint
        next_waypoint = self.get_next_waypoint(pose)
        if not next_waypoint:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel, self.local_planner_node.get_clock().now().to_msg()

        # Calculate direction to waypoint
        dx = next_waypoint.pose.position.x - pose.pose.position.x
        dy = next_waypoint.pose.position.y - pose.pose.position.y
        distance_to_waypoint = math.sqrt(dx*dx + dy*dy)

        # Check if close enough to waypoint
        if distance_to_waypoint < 0.3:  # 30cm threshold
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.current_path.poses):
                # Reached goal
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                return cmd_vel, self.local_planner_node.get_clock().now().to_msg()

            # Get new next waypoint
            next_waypoint = self.get_next_waypoint(pose)
            dx = next_waypoint.pose.position.x - pose.pose.position.x
            dy = next_waypoint.pose.position.y - pose.pose.position.y
            distance_to_waypoint = math.sqrt(dx*dx + dy*dy)

        # Calculate desired heading
        desired_theta = math.atan2(dy, dx)
        current_theta = self.get_yaw_from_pose(pose.pose.orientation)

        # Calculate angular error
        angular_error = desired_theta - current_theta
        # Normalize angle to [-pi, pi]
        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi

        # Calculate velocities
        if abs(angular_error) > 0.2:  # 0.2 rad = ~11 degrees
            # Turn in place first
            cmd_vel.angular.z = max(-self.max_angular_vel,
                                  min(self.max_angular_vel,
                                      angular_error * 0.5))
            cmd_vel.linear.x = 0.0
        else:
            # Move forward with angular correction
            cmd_vel.linear.x = max(0.0, min(self.max_linear_vel,
                                          distance_to_waypoint * 0.5))
            cmd_vel.angular.z = max(-self.max_angular_vel,
                                  min(self.max_angular_vel,
                                      angular_error * 1.0))

        # Check for obstacles in local costmap
        if self.is_path_blocked(cmd_vel):
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        return cmd_vel, self.local_planner_node.get_clock().now().to_msg()

    def get_next_waypoint(self, current_pose):
        """Get the next relevant waypoint from the path"""
        if not self.current_path or self.current_waypoint_idx >= len(self.current_path.poses):
            return None

        # Return the current waypoint we're heading to
        return self.current_path.poses[self.current_waypoint_idx]

    def get_yaw_from_pose(self, orientation):
        """Extract yaw from quaternion orientation"""
        # Convert quaternion to yaw (simplified - in practice use tf2)
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def is_path_blocked(self, cmd_vel):
        """Check if the planned path is blocked by obstacles"""
        costmap = self.costmap.get_costmap()
        # Check immediate area for obstacles
        # This is a simplified check - in practice, check along the planned path
        robot_x = int((cmd_vel.linear.x + 0.1) / costmap.resolution)  # 10cm ahead
        robot_y = int(cmd_vel.angular.z / costmap.resolution)  # Account for turning

        # Check cost at future robot position
        if (0 <= robot_x < costmap.size_x and 0 <= robot_y < costmap.size_y):
            cost = costmap.get_cost(robot_x, robot_y)
            return cost >= 200  # High cost indicates obstacle

        return False
```

## Trajectory Execution and Footstep Planning

### Footstep Planning Algorithm

```python
# footstep_planner.py
import numpy as np
import math
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class FootstepPlanner:
    def __init__(self):
        self.step_length = 0.4  # 40cm step length
        self.step_width = 0.25  # 25cm step width
        self.step_height = 0.15 # 15cm max step height
        self.balance_margin = 0.1
        self.support_polygon = []  # Points defining support polygon

    def plan_footsteps(self, start_pose, goal_pose, path):
        """Plan safe footsteps along the navigation path"""
        footsteps = []

        # Start with current pose
        left_foot = self.calculate_initial_left_foot(start_pose)
        right_foot = self.calculate_initial_right_foot(start_pose)

        current_left = left_foot
        current_right = right_foot
        support_foot = 'left'  # Start with left foot as swing foot

        # Follow the path and plan footsteps
        for i in range(1, len(path)):
            target_pos = [path[i].pose.position.x, path[i].pose.position.y]

            # Calculate next foot placement
            if support_foot == 'left':
                next_right = self.calculate_next_footstep(current_left, target_pos)
                footsteps.append(('right', next_right))
                current_right = next_right
                support_foot = 'right'
            else:
                next_left = self.calculate_next_footstep(current_right, target_pos)
                footsteps.append(('left', next_left))
                current_left = next_left
                support_foot = 'left'

        return footsteps

    def calculate_initial_left_foot(self, robot_pose):
        """Calculate initial left foot position"""
        # Offset from robot center (simplified)
        left_foot = Point()
        left_foot.x = robot_pose.pose.position.x - 0.1  # 10cm to the left
        left_foot.y = robot_pose.pose.position.y + 0.125  # Half step width
        left_foot.z = robot_pose.pose.position.z  # Same height
        return left_foot

    def calculate_initial_right_foot(self, robot_pose):
        """Calculate initial right foot position"""
        # Offset from robot center (simplified)
        right_foot = Point()
        right_foot.x = robot_pose.pose.position.x - 0.1  # 10cm to the left
        right_foot.y = robot_pose.pose.position.y - 0.125  # Half step width
        right_foot.z = robot_pose.pose.position.z  # Same height
        return right_foot

    def calculate_next_footstep(self, support_foot, target_pos):
        """Calculate next footstep position based on target"""
        next_foot = Point()

        # Calculate direction to target
        dx = target_pos[0] - support_foot.x
        dy = target_pos[1] - support_foot.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Limit step size to humanoid capabilities
        step_scale = min(1.0, self.step_length / max(distance, 0.01))

        next_foot.x = support_foot.x + dx * step_scale
        next_foot.y = support_foot.y + dy * step_scale
        next_foot.z = support_foot.z  # Maintain same height

        return next_foot

    def check_balance_feasibility(self, left_foot, right_foot, com_pos):
        """Check if CoM is within support polygon"""
        # Calculate support polygon from feet positions
        support_points = [
            [left_foot.x, left_foot.y],
            [right_foot.x, right_foot.y]
        ]

        # For bipedal stance, CoM should be between feet
        min_x = min(left_foot.x, right_foot.x)
        max_x = max(left_foot.x, right_foot.x)
        min_y = min(left_foot.y, right_foot.y) - self.balance_margin
        max_y = max(left_foot.y, right_foot.y) + self.balance_margin

        # Check if CoM is within bounds
        if (min_x - self.balance_margin <= com_pos[0] <= max_x + self.balance_margin and
            min_y <= com_pos[1] <= max_y):
            return True
        return False

    def generate_support_polygon_marker(self, left_foot, right_foot):
        """Generate visualization marker for support polygon"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Add points to form the support polygon
        points = []
        points.append(left_foot)
        points.append(right_foot)
        points.append(left_foot)  # Close the loop

        marker.points = points
        return marker
```

### Bipedal Controller

```python
# bipedal_controller.py
import numpy as np
import math
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time

class BipedalController:
    def __init__(self):
        self.step_frequency = 1.0  # 1 Hz step frequency
        self.step_height = 0.05    # 5cm step height
        self.com_height = 0.8      # 80cm CoM height
        self.foot_length = 0.2     # 20cm foot length
        self.foot_width = 0.15     # 15cm foot width

        # Walking pattern parameters
        self.stride_length = 0.4   # 40cm stride
        self.step_width = 0.25     # 25cm step width

    def generate_walking_pattern(self, velocity_cmd, current_pose):
        """Generate joint trajectories for walking based on velocity command"""
        # Convert velocity command to step parameters
        linear_vel = velocity_cmd.linear.x
        angular_vel = velocity_cmd.angular.z

        # Calculate step timing based on desired velocity
        step_duration = max(0.5, self.stride_length / max(abs(linear_vel), 0.1))

        # Generate footstep trajectories
        left_foot_traj = self.generate_foot_trajectory('left', linear_vel, angular_vel, step_duration)
        right_foot_traj = self.generate_foot_trajectory('right', linear_vel, angular_vel, step_duration)

        # Generate CoM trajectory for balance
        com_traj = self.generate_com_trajectory(linear_vel, angular_vel, step_duration)

        # Generate joint trajectories from foot and CoM trajectories
        joint_traj = self.inverse_kinematics(left_foot_traj, right_foot_traj, com_traj)

        return joint_traj

    def generate_foot_trajectory(self, foot_side, linear_vel, angular_vel, step_duration):
        """Generate foot trajectory for a single step"""
        trajectory = []

        # Calculate step parameters based on velocity
        step_length = linear_vel * step_duration
        step_width = self.step_width if foot_side == 'left' else -self.step_width

        # Generate trajectory points (simplified)
        num_points = 10
        for i in range(num_points):
            t = i / (num_points - 1)

            # Swing phase: foot moves forward and up
            if t < 0.5:
                x = step_length * t
                y = step_width
                z = self.step_height * math.sin(math.pi * t)  # Parabolic lift
            # Stance phase: foot moves down
            else:
                x = step_length * t
                y = step_width
                z = self.step_height * math.sin(math.pi * t) * (1 - (t - 0.5) * 2)

            point = Point()
            point.x = x
            point.y = y
            point.z = z
            trajectory.append(point)

        return trajectory

    def generate_com_trajectory(self, linear_vel, angular_vel, step_duration):
        """Generate CoM trajectory for balance during walking"""
        # Simplified inverted pendulum model for CoM trajectory
        trajectory = []

        num_points = 10
        for i in range(num_points):
            t = i / (num_points - 1)

            # Keep CoM at constant height for stability
            com_point = Point()
            com_point.x = linear_vel * t * step_duration
            com_point.y = 0.0  # Maintain center between feet
            com_point.z = self.com_height

            trajectory.append(com_point)

        return trajectory

    def inverse_kinematics(self, left_foot_traj, right_foot_traj, com_traj):
        """Calculate joint angles from foot and CoM positions"""
        # Simplified inverse kinematics (in practice, use robot-specific IK)
        joint_trajectories = {
            'left_leg': [],
            'right_leg': []
        }

        for i in range(len(left_foot_traj)):
            # Calculate joint angles for each trajectory point
            left_joints = self.calculate_leg_joints(left_foot_traj[i], 'left')
            right_joints = self.calculate_leg_joints(right_foot_traj[i], 'right')

            joint_trajectories['left_leg'].append(left_joints)
            joint_trajectories['right_leg'].append(right_joints)

        return joint_trajectories

    def calculate_leg_joints(self, foot_pos, leg_side):
        """Calculate joint angles for a single leg to reach foot position"""
        # Simplified 3-DOF leg model (hip, knee, ankle)
        # In practice, use full robot kinematics
        joints = Float64MultiArray()

        # Calculate approximate joint angles (simplified)
        # This would use proper IK in a real implementation
        hip_yaw = 0.0  # Simplified
        hip_pitch = math.atan2(foot_pos.z, abs(foot_pos.x) + 0.01)  # Approximate
        knee_pitch = -hip_pitch * 2  # Approximate for simple model

        joints.data = [hip_yaw, hip_pitch, knee_pitch]
        return joints
```

## Balance and Stability Considerations

### Balance Management System

```python
# balance_manager.py
import numpy as np
import math
from geometry_msgs.msg import Point, Vector3, Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool
import time

class BalanceManager:
    def __init__(self):
        self.com_position = Point()  # Center of Mass position
        self.com_velocity = Vector3()
        self.com_acceleration = Vector3()

        self.foot_positions = {
            'left': Point(),
            'right': Point()
        }

        self.balance_threshold = 0.05  # 5cm balance threshold
        self.recovery_threshold = 0.1  # 10cm recovery threshold
        self.fall_threshold = 0.2      # 20cm fall threshold

        self.imu_data = None
        self.joint_states = None

        # PID controllers for balance
        self.roll_controller = BalancePID(kp=2.0, ki=0.1, kd=0.5)
        self.pitch_controller = BalancePID(kp=2.5, ki=0.1, kd=0.6)
        self.zmp_controller = BalancePID(kp=1.5, ki=0.05, kd=0.3)

        self.is_balanced = True
        self.recovery_active = False

    def update_sensors(self, imu_data, joint_states):
        """Update sensor data for balance calculations"""
        self.imu_data = imu_data
        self.joint_states = joint_states

        # Update CoM position based on joint states (simplified)
        self.update_com_position()

    def update_com_position(self):
        """Update Center of Mass position based on joint configuration"""
        # Simplified CoM calculation (in practice, use full kinematic model)
        if self.joint_states:
            # This is a very simplified calculation
            # In practice, use robot's kinematic chain to calculate CoM
            self.com_position.x = 0.0  # Relative to base
            self.com_position.y = 0.0
            self.com_position.z = 0.8  # Approximate CoM height

    def calculate_zmp(self):
        """Calculate Zero Moment Point for balance"""
        if not self.imu_data:
            return Point()

        # Simplified ZMP calculation
        # ZMP_x = CoM_x - (CoM_z * CoM_acc_x) / gravity
        # ZMP_y = CoM_y - (CoM_z * CoM_acc_y) / gravity

        # Extract linear acceleration from IMU
        acc_x = self.imu_data.linear_acceleration.x
        acc_y = self.imu_data.linear_acceleration.y
        acc_z = self.imu_data.linear_acceleration.z

        # Gravity compensation
        gravity = 9.81
        zmp = Point()
        zmp.x = self.com_position.x - (self.com_position.z * acc_x) / gravity
        zmp.y = self.com_position.y - (self.com_position.z * acc_y) / gravity
        zmp.z = 0.0  # ZMP is on ground plane

        return zmp

    def check_balance(self):
        """Check if robot is within balance limits"""
        if not self.imu_data:
            return False

        # Calculate ZMP
        zmp = self.calculate_zmp()

        # Define support polygon (simplified as rectangle between feet)
        support_polygon = self.calculate_support_polygon()

        # Check if ZMP is within support polygon
        is_stable = self.is_point_in_polygon(zmp, support_polygon)

        # Additional checks: roll, pitch angles
        roll, pitch = self.get_orientation_angles()
        roll_ok = abs(roll) < math.radians(15)  # 15 degree limit
        pitch_ok = abs(pitch) < math.radians(15)  # 15 degree limit

        overall_stable = is_stable and roll_ok and pitch_ok

        # Update balance status
        if not overall_stable:
            if not self.is_balanced:
                # Already unstable, check for recovery/fall
                zmp_distance = self.distance_to_support_polygon(zmp, support_polygon)
                if zmp_distance > self.fall_threshold:
                    self.is_balanced = False
                    return False
                elif zmp_distance > self.recovery_threshold:
                    self.recovery_active = True
            else:
                # Just became unstable, start recovery
                self.recovery_active = True
                self.is_balanced = False
        else:
            # Stable, reset recovery if it was active
            if self.recovery_active:
                self.recovery_active = False
            self.is_balanced = True

        return self.is_balanced

    def calculate_support_polygon(self):
        """Calculate support polygon from foot positions"""
        # Simplified support polygon as rectangle between feet
        # In practice, this would be more complex based on foot geometry
        left_foot = self.foot_positions['left']
        right_foot = self.foot_positions['right']

        # Calculate bounding box of support area
        min_x = min(left_foot.x, right_foot.x) - 0.1  # Add margin
        max_x = max(left_foot.x, right_foot.x) + 0.1
        min_y = min(left_foot.y, right_foot.y) - 0.15  # Foot width margin
        max_y = max(left_foot.y, right_foot.y) + 0.15

        return [
            Point(x=min_x, y=min_y, z=0),
            Point(x=max_x, y=min_y, z=0),
            Point(x=max_x, y=max_y, z=0),
            Point(x=min_x, y=max_y, z=0)
        ]

    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting"""
        x, y = point.x, point.y
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def distance_to_support_polygon(self, point, polygon):
        """Calculate minimum distance from point to polygon edges"""
        min_distance = float('inf')

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            # Calculate distance from point to line segment
            distance = self.distance_point_to_line_segment(point, p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def distance_point_to_line_segment(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        # Vector from line_start to line_end
        line_vec = [line_end.x - line_start.x, line_end.y - line_start.y]
        point_vec = [point.x - line_start.x, point.y - line_start.y]

        line_len_sq = line_vec[0]**2 + line_vec[1]**2

        if line_len_sq == 0:
            # Line segment is actually a point
            return math.sqrt((point.x - line_start.x)**2 + (point.y - line_start.y)**2)

        # Project point_vec onto line_vec
        t = max(0, min(1, (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len_sq))

        # Find closest point on line segment
        projection = [
            line_start.x + t * line_vec[0],
            line_start.y + t * line_vec[1]
        ]

        # Return distance to closest point
        return math.sqrt((point.x - projection[0])**2 + (point.y - projection[1])**2)

    def get_orientation_angles(self):
        """Extract roll and pitch angles from IMU quaternion"""
        if not self.imu_data:
            return 0.0, 0.0

        quat = self.imu_data.orientation
        # Convert quaternion to roll, pitch, yaw (simplified)
        # Using standard conversion formulas
        sinr_cosp = 2 * (quat.w * quat.x + quat.y * quat.z)
        cosr_cosp = 1 - 2 * (quat.x * quat.x + quat.y * quat.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (quat.w * quat.y - quat.z * quat.x)
        pitch = math.asin(sinp)

        return roll, pitch

    def generate_balance_correction(self, cmd_vel):
        """Generate balance correction for navigation commands"""
        if not self.is_balanced and not self.recovery_active:
            # Generate recovery command
            return self.generate_recovery_command()

        if not self.imu_data:
            return cmd_vel

        # Get current orientation
        roll, pitch = self.get_orientation_angles()

        # Calculate balance corrections using PID controllers
        roll_correction = self.roll_controller.update(0.0, roll)  # Target roll = 0
        pitch_correction = self.pitch_controller.update(0.0, pitch)  # Target pitch = 0

        # Apply corrections to command velocity
        corrected_cmd = Twist()
        corrected_cmd.linear.x = cmd_vel.linear.x
        corrected_cmd.linear.y = cmd_vel.linear.y
        corrected_cmd.linear.z = cmd_vel.linear.z
        corrected_cmd.angular.x = cmd_vel.angular.x
        corrected_cmd.angular.y = cmd_vel.angular.y
        corrected_cmd.angular.z = cmd_vel.angular.z + roll_correction

        # Reduce forward velocity if pitch is high (for stability)
        pitch_factor = max(0.1, 1.0 - abs(pitch) / math.radians(10))
        corrected_cmd.linear.x *= pitch_factor

        return corrected_cmd

    def generate_recovery_command(self):
        """Generate command for balance recovery"""
        cmd = Twist()

        if not self.imu_data:
            # Emergency stop
            return cmd

        # Get orientation
        roll, pitch = self.get_orientation_angles()

        # Determine recovery direction based on tilt
        if abs(roll) > abs(pitch):
            # Roll is dominant, recover in roll direction
            cmd.angular.z = -np.sign(roll) * 0.3  # Counter the roll
        else:
            # Pitch is dominant, reduce forward motion
            cmd.linear.x = -np.sign(pitch) * 0.1  # Counter the pitch

        return cmd

class BalancePID:
    """Simple PID controller for balance"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, target, current):
        """Update PID and return control output"""
        error = target - current

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error
        i_term = self.ki * self.integral

        # Derivative term
        derivative = error - self.prev_error
        d_term = self.kd * derivative

        self.prev_error = error

        return p_term + i_term + d_term
```

### Integration with Navigation System

```python
# humanoid_navigator.py
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool
import math

class HumanoidNavigator(Node):
    def __init__(self):
        super().__init__('humanoid_navigator')

        # Initialize balance manager
        self.balance_manager = BalanceManager()

        # Create subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.balance_status_pub = self.create_publisher(Bool, '/balance_status', 10)

        # Timer for balance checking
        self.balance_timer = self.create_timer(0.1, self.balance_check_callback)

        self.get_logger().info('Humanoid Navigator initialized')

    def imu_callback(self, msg):
        """Handle IMU data for balance calculations"""
        self.balance_manager.update_sensors(msg, self.joint_states)

    def joint_callback(self, msg):
        """Handle joint state data"""
        self.joint_states = msg
        self.balance_manager.update_sensors(self.imu_data, msg)

    def balance_check_callback(self):
        """Periodic balance check"""
        is_balanced = self.balance_manager.check_balance()

        # Publish balance status
        balance_msg = Bool()
        balance_msg.data = is_balanced
        self.balance_status_pub.publish(balance_msg)

    def execute_navigation(self, goal_pose):
        """Execute navigation with balance integration"""
        # This would integrate with Nav2's action server
        # For now, showing the concept

        # Plan path using humanoid-aware planner
        path = self.plan_humanoid_path(goal_pose)

        if not path:
            self.get_logger().error('Failed to plan path for humanoid')
            return False

        # Execute navigation with balance monitoring
        success = self.follow_path_with_balance(path)
        return success

    def plan_humanoid_path(self, goal_pose):
        """Plan path considering humanoid constraints"""
        # This would call the humanoid global planner
        # Implementation would use the HumanoidGlobalPlanner class
        return []  # Placeholder

    def follow_path_with_balance(self, path):
        """Follow path while maintaining balance"""
        # This would integrate with the local planner and controller
        # Implementation would use the HumanoidLocalPlanner and BipedalController
        return True  # Placeholder
```

## Perception Integration

### Terrain Analysis for Humanoid Navigation

```python
# terrain_analyzer.py
import numpy as np
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import math

class TerrainAnalyzer:
    def __init__(self):
        self.step_height_threshold = 0.15  # 15cm max step height
        self.gap_threshold = 0.30          # 30cm max gap width
        self.slope_threshold = math.radians(20)  # 20 degree max slope
        self.traversable_threshold = 0.10  # 10cm max irregularity

    def analyze_pointcloud(self, pointcloud):
        """Analyze point cloud data for traversable terrain"""
        # Convert point cloud to organized structure
        points = self.pointcloud_to_array(pointcloud)

        # Analyze terrain characteristics
        traversable_map = self.analyze_terrain_points(points)

        return traversable_map

    def pointcloud_to_array(self, pointcloud):
        """Convert PointCloud2 message to numpy array"""
        # Implementation would convert PointCloud2 to array of points
        # This is a simplified representation
        points = []
        # In practice, use sensor_msgs.point_cloud2.read_points
        return points

    def analyze_terrain_points(self, points):
        """Analyze terrain points for humanoid traversability"""
        traversable_map = {}

        for point in points:
            # Check if point is ground level (not obstacle)
            if self.is_ground_point(point, points):
                # Analyze local terrain around this point
                terrain_info = self.analyze_local_terrain(point, points)

                # Determine traversability
                is_traversable = self.is_traversable(terrain_info)
                traversable_map[point] = is_traversable

        return traversable_map

    def is_ground_point(self, point, all_points):
        """Determine if a point is part of the ground plane"""
        # Use RANSAC or other plane fitting algorithms
        # to identify ground points
        return True  # Simplified

    def analyze_local_terrain(self, center_point, all_points):
        """Analyze terrain characteristics around a point"""
        # Get neighboring points
        neighbors = self.get_neighbors(center_point, all_points, radius=0.5)

        # Calculate terrain properties
        height_variance = self.calculate_height_variance(neighbors)
        slope = self.calculate_slope(neighbors)
        step_height = self.calculate_step_height(neighbors)

        return {
            'height_variance': height_variance,
            'slope': slope,
            'step_height': step_height,
            'is_flat': height_variance < self.traversable_threshold,
            'is_steep': slope > self.slope_threshold,
            'has_steps': step_height > self.step_height_threshold
        }

    def get_neighbors(self, center_point, all_points, radius):
        """Get neighboring points within a radius"""
        neighbors = []
        for point in all_points:
            distance = self.calculate_distance(center_point, point)
            if distance <= radius:
                neighbors.append(point)
        return neighbors

    def calculate_distance(self, point1, point2):
        """Calculate 3D distance between two points"""
        dx = point1.x - point2.x
        dy = point1.y - point2.y
        dz = point1.z - point2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def calculate_height_variance(self, points):
        """Calculate height variance in a local area"""
        if not points:
            return 0.0

        heights = [p.z for p in points]
        mean_height = sum(heights) / len(heights)
        variance = sum((h - mean_height)**2 for h in heights) / len(heights)
        return math.sqrt(variance)

    def calculate_slope(self, points):
        """Calculate local terrain slope"""
        if len(points) < 3:
            return 0.0

        # Fit plane to points and calculate slope
        # Simplified approach: use height differences
        min_z = min(p.z for p in points)
        max_z = max(p.z for p in points)

        # Calculate approximate slope over the area
        area_size = len(points) * 0.01  # Approximate area factor
        return math.atan2(max_z - min_z, area_size)

    def calculate_step_height(self, points):
        """Calculate maximum step height in the area"""
        if not points:
            return 0.0

        z_values = sorted([p.z for p in points])
        if len(z_values) < 2:
            return 0.0

        # Find largest step between adjacent height levels
        max_step = 0.0
        for i in range(1, len(z_values)):
            step = z_values[i] - z_values[i-1]
            max_step = max(max_step, step)

        return max_step

    def is_traversable(self, terrain_info):
        """Determine if terrain is traversable for humanoid"""
        if terrain_info['has_steps'] and terrain_info['step_height'] > self.step_height_threshold:
            return False

        if terrain_info['is_steep'] and terrain_info['slope'] > self.slope_threshold:
            return False

        if not terrain_info['is_flat'] and terrain_info['height_variance'] > self.traversable_threshold:
            return False

        return True
```

## Recovery Behaviors

### Humanoid-Specific Recovery Behaviors

```python
# humanoid_recovery_behaviors.py
import rclpy
from rclpy.node import Node
from nav2_core.recovery import Recovery
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
import math

class BalanceRecovery(Recovery):
    def __init__(self):
        super().__init__()
        self.recovery_cmd = Twist()
        self.balance_threshold = 0.1
        self.max_recovery_time = 5.0  # 5 seconds max recovery time

    def on_configure(self, recovery_node):
        """Configure the recovery behavior"""
        self.node = recovery_node
        self.logger = recovery_node.get_logger()

    def on_cleanup(self):
        """Clean up the recovery behavior"""
        pass

    def on_activate(self):
        """Activate the recovery behavior"""
        pass

    def on_deactivate(self):
        """Deactivate the recovery behavior"""
        pass

    def run(self, original_command):
        """Execute the balance recovery behavior"""
        self.logger.info('Executing balance recovery behavior')

        start_time = self.node.get_clock().now()

        # Attempt to recover balance
        while rclpy.ok():
            current_time = self.node.get_clock().now()
            elapsed_time = (current_time - start_time).nanoseconds / 1e9

            if elapsed_time > self.max_recovery_time:
                self.logger.warn('Balance recovery timed out')
                return False

            # Check if balance is recovered
            if self.is_balanced():
                self.logger.info('Balance recovered successfully')
                return True

            # Generate recovery command
            recovery_cmd = self.generate_balance_recovery_command()
            self.publish_cmd_vel(recovery_cmd)

            # Small delay
            time.sleep(0.1)

        return False

    def is_balanced(self):
        """Check if robot is currently balanced"""
        # This would check balance status from balance manager
        # For now, return a placeholder
        return False

    def generate_balance_recovery_command(self):
        """Generate command to recover balance"""
        cmd = Twist()

        # Implement balance recovery logic
        # This would analyze current state and generate appropriate command
        cmd.linear.x = 0.0  # Stop forward motion
        cmd.angular.z = 0.0  # Stop turning

        return cmd

    def publish_cmd_vel(self, cmd_vel):
        """Publish velocity command"""
        # This would publish to the robot's velocity command topic
        pass

class SafeStopRecovery(Recovery):
    def __init__(self):
        super().__init__()
        self.stop_cmd = Twist()

    def on_configure(self, recovery_node):
        """Configure the recovery behavior"""
        self.node = recovery_node
        self.logger = recovery_node.get_logger()

    def on_cleanup(self):
        """Clean up the recovery behavior"""
        pass

    def on_activate(self):
        """Activate the recovery behavior"""
        pass

    def on_deactivate(self):
        """Deactivate the recovery behavior"""
        pass

    def run(self, original_command):
        """Execute the safe stop recovery behavior"""
        self.logger.info('Executing safe stop recovery behavior')

        # Generate safe stop command
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0

        # Publish stop command
        self.publish_cmd_vel(stop_cmd)

        # Wait briefly to ensure robot stops
        time.sleep(0.5)

        # Check if robot has stopped
        if self.is_robot_stopped():
            self.logger.info('Robot stopped safely')
            return True
        else:
            self.logger.warn('Robot may not have stopped completely')
            return False

    def is_robot_stopped(self):
        """Check if robot has stopped moving"""
        # This would check robot's current velocity
        return True

    def publish_cmd_vel(self, cmd_vel):
        """Publish velocity command"""
        # This would publish to the robot's velocity command topic
        pass
```

## Testing and Validation

### Unit Tests for Humanoid Navigation Components

```python
# test_humanoid_nav.py
import unittest
import numpy as np
from geometry_msgs.msg import Pose, Point
from humanoid_global_planner import HumanoidGlobalPlanner
from balance_manager import BalanceManager

class TestHumanoidGlobalPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = HumanoidGlobalPlanner()

    def test_valid_pose_check(self):
        """Test pose validation for humanoid navigation"""
        # Create a valid pose
        pose = Pose()
        pose.position.x = 1.0
        pose.position.y = 1.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0

        # This would test against a costmap, simplified here
        self.assertTrue(True)  # Placeholder

    def test_terrain_traversability(self):
        """Test terrain traversability check"""
        start = Pose()
        start.position.x = 0.0
        start.position.y = 0.0

        goal = Pose()
        goal.position.x = 5.0
        goal.position.y = 0.0

        # This would test terrain analysis, simplified here
        self.assertTrue(True)  # Placeholder

class TestBalanceManager(unittest.TestCase):
    def setUp(self):
        self.balance_manager = BalanceManager()

    def test_zmp_calculation(self):
        """Test Zero Moment Point calculation"""
        # Simulate IMU data
        class MockIMU:
            def __init__(self):
                self.linear_acceleration = Point()
                self.linear_acceleration.x = 0.5
                self.linear_acceleration.y = 0.2
                self.linear_acceleration.z = 9.81  # Gravity

                self.orientation = Pose()
                self.orientation.w = 1.0

        self.balance_manager.imu_data = MockIMU()
        self.balance_manager.com_position.z = 0.8  # 80cm CoM height

        zmp = self.balance_manager.calculate_zmp()

        # ZMP should be calculable
        self.assertIsNotNone(zmp)

    def test_balance_check(self):
        """Test balance checking functionality"""
        # This would test the complete balance checking system
        self.assertTrue(True)  # Placeholder

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
# integration_test_humanoid_nav.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import time

class HumanoidNavIntegrationTest(Node):
    def __init__(self):
        super().__init__('humanoid_nav_integration_test')

        # Create action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def test_simple_navigation(self):
        """Test simple navigation with humanoid constraints"""
        # Wait for action server
        self.nav_client.wait_for_server()

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = 2.0
        goal_msg.pose.pose.position.y = 2.0
        goal_msg.pose.pose.orientation.w = 1.0

        # Send goal
        goal_future = self.nav_client.send_goal_async(goal_msg)

        # Wait for result
        rclpy.spin_until_future_complete(self, goal_future)

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f'Navigation result: {result}')

        return True

    def test_balance_integration(self):
        """Test navigation with balance system integration"""
        # This would test the complete system integration
        # including balance monitoring during navigation
        pass

def main():
    rclpy.init()

    test_node = HumanoidNavIntegrationTest()

    # Run tests
    success = test_node.test_simple_navigation()

    if success:
        print("Integration test passed!")
    else:
        print("Integration test failed!")

    test_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary and Next Steps

### Key Takeaways

1. **Humanoid-Specific Navigation**: Humanoid robots require specialized navigation approaches that consider balance, footstep planning, and terrain constraints.

2. **Balance Management**: Maintaining dynamic balance during navigation is crucial and requires real-time monitoring and correction.

3. **Footstep Planning**: Safe and stable foot placement is essential for humanoid locomotion and navigation.

4. **Perception Integration**: 3D perception and terrain analysis are critical for humanoid navigation in complex environments.

5. **Recovery Behaviors**: Specialized recovery behaviors are needed to handle balance losses and ensure safe operation.

### Next Steps

1. **Implementation**: Implement the humanoid navigation system on a physical or simulated humanoid robot.

2. **Tuning**: Fine-tune parameters based on the specific robot's physical characteristics and capabilities.

3. **Testing**: Conduct extensive testing in various environments to validate the navigation system.

4. **Optimization**: Optimize the system for real-time performance and energy efficiency.

5. **Integration**: Integrate with higher-level planning and AI systems for autonomous operation.

### Advanced Topics

For further development, consider:
- Machine learning approaches for adaptive gait and balance control
- Multi-modal perception fusion for robust environment understanding
- Learning from demonstration for improved navigation strategies
- Human-aware navigation for collaborative environments

## Exercises

1. Implement a humanoid-specific costmap layer that considers terrain traversability
2. Create a footstep planner that works with uneven terrain
3. Develop a balance recovery behavior for handling unexpected disturbances
4. Integrate visual SLAM with humanoid navigation for improved localization
5. Create a simulation environment to test humanoid navigation capabilities

## References

- Kajita, S. (2005). *Humanoid Robot*. MIT Press.
- Siciliano, B., & Khatib, O. (2016). *Springer Handbook of Robotics*. Springer.
- Nav2 Documentation: https://navigation.ros.org/
- ROS 2 Documentation: https://docs.ros.org/en/humble/