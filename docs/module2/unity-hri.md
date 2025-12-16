---
sidebar_position: 4
---

# Unity for Human-Robot Interaction

## Overview

This chapter introduces Unity as a platform for creating intuitive human-robot interaction (HRI) interfaces. Unity provides powerful 3D visualization capabilities that can enhance robot teleoperation, monitoring, and control. You'll learn how to create user-friendly interfaces that bridge the gap between humans and robots, making robotic systems more accessible and easier to operate.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the role of Unity in human-robot interaction
- Create basic Unity scenes for robot visualization
- Implement teleoperation interfaces using Unity
- Design intuitive control panels for robot operation
- Integrate Unity with ROS 2 for real-time data exchange
- Develop monitoring dashboards for robot status

## Table of Contents

1. [Introduction to Unity for HRI](#introduction-to-unity-for-hri)
2. [Setting Up Unity for Robotics](#setting-up-unity-for-robotics)
3. [Basic Robot Visualization](#basic-robot-visualization)
4. [Teleoperation Interfaces](#teleoperation-interfaces)
5. [Control Panel Design](#control-panel-design)
6. [ROS 2 Integration](#ros-2-integration)
7. [Monitoring and Visualization](#monitoring-and-visualization)
8. [User Experience Considerations](#user-experience-considerations)
9. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to Unity for HRI

### What is Human-Robot Interaction?

Human-Robot Interaction (HRI) is a multidisciplinary field focused on understanding and designing the interactions between humans and robots. In robotics applications, HRI interfaces serve as the bridge between human operators and robotic systems, enabling:

- **Teleoperation**: Remote control of robots
- **Monitoring**: Real-time visualization of robot status
- **Programming**: High-level task specification
- **Supervision**: Oversight of autonomous robot behaviors

### Why Use Unity for HRI?

Unity offers several advantages for HRI applications:

- **3D Visualization**: High-quality real-time 3D rendering
- **Cross-Platform**: Deploy to multiple platforms (Windows, macOS, mobile, VR)
- **User Interface Tools**: Robust UI system for control panels
- **Asset Store**: Extensive library of 3D models and tools
- **Physics Engine**: Built-in physics for simulation
- **Scripting**: C# scripting for custom functionality
- **VR/AR Support**: Immersive interaction possibilities

### Common HRI Scenarios

1. **Remote Teleoperation**: Controlling robots in hazardous environments
2. **Supervisory Control**: Overseeing autonomous robot missions
3. **Training**: Teaching robots new behaviors
4. **Maintenance**: Diagnosing and repairing robot systems
5. **Collaboration**: Working alongside robots in shared spaces

## Setting Up Unity for Robotics

### Installing Unity

1. Download Unity Hub from the Unity website
2. Install Unity Editor (recommended version 2021.3 LTS or later)
3. Create a Unity account if you don't have one
4. Install the Universal Render Pipeline (URP) package for better performance

### Essential Unity Concepts for Robotics

#### GameObjects and Components
In Unity, everything is a GameObject with Components:

```csharp
// Example: Robot controller script
using UnityEngine;

public class RobotController : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public float rotateSpeed = 100.0f;

    void Update()
    {
        // Handle input for robot movement
        float moveInput = Input.GetAxis("Vertical");
        float turnInput = Input.GetAxis("Horizontal");

        // Move the robot
        transform.Translate(Vector3.forward * moveInput * moveSpeed * Time.deltaTime);
        transform.Rotate(Vector3.up, turnInput * rotateSpeed * Time.deltaTime);
    }
}
```

#### Coordinate Systems
Unity uses a left-handed coordinate system:
- X: Right
- Y: Up
- Z: Forward

ROS uses a right-handed coordinate system:
- X: Forward
- Y: Left
- Z: Up

### Basic Unity Scene Setup

```csharp
// RobotVisualization.cs - Basic robot visualization script
using UnityEngine;

public class RobotVisualization : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotModel;  // The 3D model of the robot
    public Transform[] jointTransforms;  // Array of joint transforms
    public float[] jointPositions;  // Current joint positions

    [Header("ROS Integration")]
    public bool useROS = true;

    void Start()
    {
        InitializeRobot();
    }

    void Update()
    {
        if (useROS)
        {
            UpdateFromROSData();
        }
        else
        {
            // For simulation/testing
            SimulateRobotMovement();
        }
    }

    void InitializeRobot()
    {
        // Initialize joint positions array
        if (jointTransforms != null)
        {
            jointPositions = new float[jointTransforms.Length];
        }
    }

    void UpdateFromROSData()
    {
        // This would be connected to ROS topic subscribers
        // For now, we'll simulate the data
        SimulateRobotMovement();
    }

    void SimulateRobotMovement()
    {
        // Simulate some basic movement for demonstration
        transform.position += Vector3.forward * Mathf.Sin(Time.time) * 0.01f;
    }
}
```

## Basic Robot Visualization

### Creating Robot Models

Unity can import robot models from various formats:
- **URDF**: Use ROS# or similar packages to convert
- **STL/OBJ**: Direct import for static models
- **FBX**: For animated models with joints
- **GLTF**: Modern format with good tooling support

### Importing and Setting Up Robot Models

1. **Import the model**: Place your robot model in the Assets folder
2. **Configure import settings**:
   - Set scale appropriately (often 1 unit = 1 meter)
   - Enable "Read/Write Enabled" if you need to modify meshes at runtime
   - Set appropriate materials and textures

3. **Create a robot controller**:

```csharp
// RobotModelController.cs
using UnityEngine;

public class RobotModelController : MonoBehaviour
{
    [System.Serializable]
    public class JointConfig
    {
        public string jointName;
        public Transform jointTransform;
        public JointType jointType;
        [Range(-180, 180)] public float minAngle = -90;
        [Range(-180, 180)] public float maxAngle = 90;
        public float currentAngle;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    public JointConfig[] joints;

    void Start()
    {
        UpdateAllJoints();
    }

    public void SetJointPosition(int jointIndex, float angle)
    {
        if (jointIndex >= 0 && jointIndex < joints.Length)
        {
            JointConfig joint = joints[jointIndex];
            joint.currentAngle = Mathf.Clamp(angle, joint.minAngle, joint.maxAngle);

            // Apply rotation based on joint type
            switch (joint.jointType)
            {
                case JointType.Revolute:
                    joint.jointTransform.localRotation = Quaternion.Euler(
                        joint.jointTransform.localRotation.eulerAngles.x,
                        joint.currentAngle,
                        joint.jointTransform.localRotation.eulerAngles.z
                    );
                    break;
                case JointType.Prismatic:
                    // For prismatic joints, you might translate along an axis
                    Vector3 newPosition = joint.jointTransform.localPosition;
                    newPosition.y = joint.currentAngle / 100f; // Scale as needed
                    joint.jointTransform.localPosition = newPosition;
                    break;
            }
        }
    }

    public void SetAllJointPositions(float[] angles)
    {
        for (int i = 0; i < Mathf.Min(joints.Length, angles.Length); i++)
        {
            SetJointPosition(i, angles[i]);
        }
    }

    void UpdateAllJoints()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            SetJointPosition(i, joints[i].currentAngle);
        }
    }
}
```

### Camera Setup for Robot Visualization

```csharp
// RobotCameraController.cs
using UnityEngine;

public class RobotCameraController : MonoBehaviour
{
    public Transform target;  // Robot to follow
    public float distance = 10.0f;
    public float height = 5.0f;
    public float smoothSpeed = 12.0f;

    public float rotationSpeed = 100.0f;
    private float currentRotationX = 0f;

    void LateUpdate()
    {
        if (target == null) return;

        // Calculate desired position
        Vector3 desiredPosition = target.position - Vector3.forward * distance + Vector3.up * height;
        Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);
        transform.position = smoothedPosition;

        // Look at the target
        transform.LookAt(target);
    }

    void Update()
    {
        // Allow manual rotation with mouse
        if (Input.GetMouseButton(1)) // Right mouse button
        {
            float mouseX = Input.GetAxis("Mouse X");
            float mouseY = Input.GetAxis("Mouse Y");

            currentRotationX -= mouseY * rotationSpeed * Time.deltaTime;
            currentRotationX = Mathf.Clamp(currentRotationX, -80f, 80f);

            transform.Rotate(Vector3.right, -mouseY * rotationSpeed * Time.deltaTime);
            transform.Rotate(Vector3.up, mouseX * rotationSpeed * Time.deltaTime, Space.World);
        }
    }
}
```

## Teleoperation Interfaces

### Basic Teleoperation Controls

```csharp
// TeleoperationController.cs
using UnityEngine;
using UnityEngine.UI;

public class TeleoperationController : MonoBehaviour
{
    [Header("Robot Connection")]
    public RobotModelController robotController;

    [Header("Control UI")]
    public Slider linearSpeedSlider;
    public Slider angularSpeedSlider;
    public Button forwardButton;
    public Button backwardButton;
    public Button leftButton;
    public Button rightButton;
    public Button stopButton;

    [Header("Speed Settings")]
    public float maxLinearSpeed = 2.0f;
    public float maxAngularSpeed = 1.0f;

    private float currentLinearSpeed = 0f;
    private float currentAngularSpeed = 0f;
    private bool isConnected = false;

    void Start()
    {
        SetupUI();
        ConnectToRobot();
    }

    void SetupUI()
    {
        if (linearSpeedSlider != null)
        {
            linearSpeedSlider.minValue = 0f;
            linearSpeedSlider.maxValue = maxLinearSpeed;
            linearSpeedSlider.value = maxLinearSpeed / 2f;
        }

        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.minValue = 0f;
            angularSpeedSlider.maxValue = maxAngularSpeed;
            angularSpeedSlider.value = maxAngularSpeed / 2f;
        }

        // Setup button events
        if (forwardButton != null) forwardButton.onClick.AddListener(() => MoveRobot(1, 0));
        if (backwardButton != null) backwardButton.onClick.AddListener(() => MoveRobot(-1, 0));
        if (leftButton != null) leftButton.onClick.AddListener(() => MoveRobot(0, 1));
        if (rightButton != null) rightButton.onClick.AddListener(() => MoveRobot(0, -1));
        if (stopButton != null) stopButton.onClick.AddListener(StopRobot);
    }

    void ConnectToRobot()
    {
        // In a real implementation, this would connect to ROS
        // For now, we'll just assume connection is successful
        isConnected = (robotController != null);

        if (isConnected)
        {
            Debug.Log("Connected to robot successfully!");
        }
        else
        {
            Debug.LogWarning("Could not connect to robot. Using simulation mode.");
        }
    }

    public void MoveRobot(int linearDir, int angularDir)
    {
        if (!isConnected) return;

        currentLinearSpeed = linearDir * linearSpeedSlider.value;
        currentAngularSpeed = angularDir * angularSpeedSlider.value;

        // Send command to robot (in real implementation)
        SendRobotCommand(currentLinearSpeed, currentAngularSpeed);
    }

    public void StopRobot()
    {
        currentLinearSpeed = 0f;
        currentAngularSpeed = 0f;
        SendRobotCommand(0f, 0f);
    }

    void SendRobotCommand(float linear, float angular)
    {
        // In a real implementation, this would send commands to ROS
        // For now, we'll just log the command
        Debug.Log($"Sending command - Linear: {linear:F2}, Angular: {angular:F2}");

        // If in simulation mode, update the robot model directly
        if (!isConnected && robotController != null)
        {
            SimulateRobotMovement(linear, angular);
        }
    }

    void SimulateRobotMovement(float linear, float angular)
    {
        // Simple simulation of robot movement
        Transform robotTransform = robotController.transform;
        robotTransform.Translate(Vector3.forward * linear * Time.deltaTime);
        robotTransform.Rotate(Vector3.up, angular * Time.deltaTime * 50f); // Scale angular for visibility
    }
}
```

### Advanced Teleoperation Features

```csharp
// AdvancedTeleoperation.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class AdvancedTeleoperation : MonoBehaviour
{
    [Header("Navigation")]
    public LayerMask groundLayer;
    public GameObject navigationMarker;
    public Button navToPointButton;

    [Header("Robot Status")]
    public Text statusText;
    public Image batteryBar;
    public Text batteryText;

    [Header("Sensor Visualization")]
    public GameObject lidarVisualization;
    public GameObject cameraFeed;

    private Vector3? navigationTarget = null;
    private float batteryLevel = 100f;

    void Start()
    {
        SetupAdvancedControls();
    }

    void SetupAdvancedControls()
    {
        if (navToPointButton != null)
        {
            navToPointButton.onClick.AddListener(SetNavigationTarget);
        }
    }

    void Update()
    {
        HandleMouseNavigation();
        UpdateRobotStatus();
    }

    void HandleMouseNavigation()
    {
        if (Input.GetMouseButtonDown(0) && !IsPointerOverUI()) // Left click
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, Mathf.Infinity, groundLayer))
            {
                // Place navigation marker
                if (navigationMarker != null)
                {
                    navigationMarker.SetActive(true);
                    navigationMarker.transform.position = hit.point;
                }

                navigationTarget = hit.point;
                Debug.Log($"Navigation target set to: {hit.point}");
            }
        }
    }

    bool IsPointerOverUI()
    {
        // This would check if the mouse is over UI elements
        // Implementation depends on your UI system
        return false;
    }

    void SetNavigationTarget()
    {
        if (navigationTarget.HasValue)
        {
            // Send navigation command to robot
            SendNavigationCommand(navigationTarget.Value);
        }
    }

    void SendNavigationCommand(Vector3 target)
    {
        Debug.Log($"Sending navigation command to: {target}");
        // In real implementation, send this to ROS navigation stack
    }

    void UpdateRobotStatus()
    {
        // Update battery visualization
        if (batteryBar != null)
        {
            batteryBar.fillAmount = batteryLevel / 100f;
        }

        if (batteryText != null)
        {
            batteryText.text = $"Battery: {batteryLevel:F1}%";
        }

        // Simulate battery drain
        if (batteryLevel > 0)
        {
            batteryLevel -= Time.deltaTime * 0.01f; // Very slow drain for demo
        }

        // Update status text
        if (statusText != null)
        {
            statusText.text = "Status: Operational";
        }
    }

    // Methods for sensor data visualization
    public void UpdateLidarData(float[] ranges, float[] angles)
    {
        // Update LiDAR visualization
        if (lidarVisualization != null)
        {
            // This would update the LiDAR visualization based on ranges and angles
            // Implementation depends on your visualization approach
        }
    }

    public void UpdateCameraFeed(Texture2D image)
    {
        // Update camera feed visualization
        if (cameraFeed != null)
        {
            // This would update the camera feed texture
            RawImage rawImage = cameraFeed.GetComponent<RawImage>();
            if (rawImage != null)
            {
                rawImage.texture = image;
            }
        }
    }
}
```

## Control Panel Design

### Creating Intuitive Control Panels

```csharp
// ControlPanelManager.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class ControlPanelManager : MonoBehaviour
{
    [Header("Control Panels")]
    public GameObject mainControlPanel;
    public GameObject navigationPanel;
    public GameObject manipulationPanel;
    public GameObject settingsPanel;

    [Header("Navigation Elements")]
    public Button navPanelButton;
    public Button manipPanelButton;
    public Button settingsPanelButton;

    [Header("Emergency Controls")]
    public Button emergencyStopButton;
    public Button resetSystemButton;

    [Header("Mode Selection")]
    public Toggle autonomousModeToggle;
    public Toggle teleopModeToggle;
    public Toggle manualModeToggle;

    private Dictionary<string, GameObject> panels;
    private string currentPanel = "main";

    void Start()
    {
        InitializePanels();
        SetupControlEvents();
    }

    void InitializePanels()
    {
        panels = new Dictionary<string, GameObject>
        {
            {"main", mainControlPanel},
            {"navigation", navigationPanel},
            {"manipulation", manipulationPanel},
            {"settings", settingsPanel}
        };

        // Hide all panels except main
        foreach (var panel in panels.Values)
        {
            if (panel != mainControlPanel)
            {
                panel.SetActive(false);
            }
        }
    }

    void SetupControlEvents()
    {
        // Panel navigation
        if (navPanelButton != null)
            navPanelButton.onClick.AddListener(() => SwitchPanel("navigation"));

        if (manipPanelButton != null)
            manipPanelButton.onClick.AddListener(() => SwitchPanel("manipulation"));

        if (settingsPanelButton != null)
            settingsPanelButton.onClick.AddListener(() => SwitchPanel("settings"));

        // Emergency controls
        if (emergencyStopButton != null)
            emergencyStopButton.onClick.AddListener(EmergencyStop);

        if (resetSystemButton != null)
            resetSystemButton.onClick.AddListener(ResetSystem);

        // Mode selection
        if (autonomousModeToggle != null)
            autonomousModeToggle.onValueChanged.AddListener((isOn) => { if (isOn) SetControlMode("autonomous"); });

        if (teleopModeToggle != null)
            teleopModeToggle.onValueChanged.AddListener((isOn) => { if (isOn) SetControlMode("teleop"); });

        if (manualModeToggle != null)
            manualModeToggle.onValueChanged.AddListener((isOn) => { if (isOn) SetControlMode("manual"); });
    }

    void SwitchPanel(string panelName)
    {
        // Hide current panel
        if (panels.ContainsKey(currentPanel))
        {
            panels[currentPanel].SetActive(false);
        }

        // Show new panel
        if (panels.ContainsKey(panelName))
        {
            panels[panelName].SetActive(true);
            currentPanel = panelName;
        }
    }

    void EmergencyStop()
    {
        Debug.LogWarning("EMERGENCY STOP ACTIVATED!");
        // In real implementation, send emergency stop to robot
        // Reset all control inputs
    }

    void ResetSystem()
    {
        Debug.Log("System reset initiated");
        // Reset system to safe state
        SwitchPanel("main");
    }

    void SetControlMode(string mode)
    {
        Debug.Log($"Control mode changed to: {mode}");
        // Handle mode change (send to robot, update UI, etc.)
    }
}
```

### Dashboard Design

```csharp
// DashboardManager.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class DashboardManager : MonoBehaviour
{
    [Header("System Status")]
    public Text systemStatusText;
    public Image systemStatusIndicator;
    public Color operationalColor = Color.green;
    public Color warningColor = Color.yellow;
    public Color errorColor = Color.red;

    [Header("Robot Status")]
    public Text robotNameText;
    public Text positionText;
    public Text batteryText;
    public Text speedText;

    [Header("Sensor Status")]
    public List<SensorStatusDisplay> sensorDisplays;

    [Header("Performance Metrics")]
    public Text cpuUsageText;
    public Text memoryUsageText;
    public Text networkLatencyText;

    [Header("Mission Information")]
    public Text missionStatusText;
    public Text taskProgressText;
    public Slider taskProgressSlider;

    private Dictionary<string, SystemStatus> sensorStatuses;

    void Start()
    {
        InitializeDashboard();
        UpdateDashboard();
    }

    void InitializeDashboard()
    {
        sensorStatuses = new Dictionary<string, SystemStatus>();

        // Initialize sensor displays
        foreach (var sensorDisplay in sensorDisplays)
        {
            sensorStatuses[sensorDisplay.sensorName] = SystemStatus.Unknown;
        }

        // Set initial robot name
        if (robotNameText != null)
        {
            robotNameText.text = "Robot-001";
        }
    }

    void Update()
    {
        UpdateDashboard();
    }

    void UpdateDashboard()
    {
        // Update system status
        SystemStatus overallStatus = GetOverallSystemStatus();
        UpdateSystemStatus(overallStatus);

        // Update robot status
        UpdateRobotStatus();

        // Update sensor statuses
        UpdateSensorStatuses();

        // Update performance metrics
        UpdatePerformanceMetrics();

        // Update mission information
        UpdateMissionInfo();
    }

    SystemStatus GetOverallSystemStatus()
    {
        // Determine overall system status based on all components
        bool hasError = false;
        bool hasWarning = false;

        foreach (var status in sensorStatuses.Values)
        {
            if (status == SystemStatus.Error) hasError = true;
            if (status == SystemStatus.Warning) hasWarning = true;
        }

        if (hasError) return SystemStatus.Error;
        if (hasWarning) return SystemStatus.Warning;
        return SystemStatus.Operational;
    }

    void UpdateSystemStatus(SystemStatus status)
    {
        if (systemStatusText != null)
        {
            systemStatusText.text = status.ToString();
        }

        if (systemStatusIndicator != null)
        {
            switch (status)
            {
                case SystemStatus.Operational:
                    systemStatusIndicator.color = operationalColor;
                    break;
                case SystemStatus.Warning:
                    systemStatusIndicator.color = warningColor;
                    break;
                case SystemStatus.Error:
                    systemStatusIndicator.color = errorColor;
                    break;
            }
        }
    }

    void UpdateRobotStatus()
    {
        // Simulate updating robot status
        if (positionText != null)
        {
            positionText.text = $"Position: {transform.position.x:F2}, {transform.position.y:F2}, {transform.position.z:F2}";
        }

        if (batteryText != null)
        {
            batteryText.text = "Battery: 87%";
        }

        if (speedText != null)
        {
            speedText.text = "Speed: 0.5 m/s";
        }
    }

    void UpdateSensorStatuses()
    {
        // Update each sensor display
        foreach (var sensorDisplay in sensorDisplays)
        {
            if (sensorStatuses.ContainsKey(sensorDisplay.sensorName))
            {
                sensorDisplay.UpdateStatus(sensorStatuses[sensorDisplay.sensorName]);
            }
        }
    }

    void UpdatePerformanceMetrics()
    {
        // In a real implementation, these would come from system monitoring
        if (cpuUsageText != null) cpuUsageText.text = "CPU: 45%";
        if (memoryUsageText != null) memoryUsageText.text = "Memory: 62%";
        if (networkLatencyText != null) networkLatencyText.text = "Latency: 42ms";
    }

    void UpdateMissionInfo()
    {
        if (missionStatusText != null) missionStatusText.text = "Mission: Patrol Route A";
        if (taskProgressText != null) taskProgressText.text = "Progress: 65%";
        if (taskProgressSlider != null) taskProgressSlider.value = 0.65f;
    }
}

// Supporting class for sensor status display
[System.Serializable]
public class SensorStatusDisplay
{
    public string sensorName;
    public Text statusText;
    public Image statusIndicator;

    public void UpdateStatus(SystemStatus status)
    {
        if (statusText != null)
        {
            statusText.text = $"{sensorName}: {status}";
        }

        if (statusIndicator != null)
        {
            switch (status)
            {
                case SystemStatus.Operational:
                    statusIndicator.color = Color.green;
                    break;
                case SystemStatus.Warning:
                    statusIndicator.color = Color.yellow;
                    break;
                case SystemStatus.Error:
                    statusIndicator.color = Color.red;
                    break;
                default:
                    statusIndicator.color = Color.gray;
                    break;
            }
        }
    }
}

public enum SystemStatus
{
    Unknown,
    Operational,
    Warning,
    Error
}
```

## ROS 2 Integration

### Unity-Rosbridge Integration

While Unity doesn't have native ROS 2 support, you can integrate using rosbridge_suite and WebSockets:

1. **Install rosbridge_suite**:
```bash
sudo apt-get install ros-humble-rosbridge-suite
```

2. **Launch rosbridge**:
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

3. **Use a Unity WebSocket library** to communicate with rosbridge.

### Basic ROS Bridge Integration

```csharp
// RosBridgeClient.cs (Conceptual - would need actual WebSocket implementation)
using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;  // You'd need to import a JSON library

public class RosBridgeClient : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosbridgeUrl = "ws://localhost:9090";
    public string robotNamespace = "/my_robot";

    private bool isConnected = false;

    void Start()
    {
        ConnectToRosbridge();
    }

    void ConnectToRosbridge()
    {
        // This is conceptual - you'd need to implement WebSocket connection
        // using a library like WebSocketSharp or similar
        Debug.Log($"Connecting to ROS bridge at {rosbridgeUrl}");

        // In real implementation:
        // 1. Establish WebSocket connection
        // 2. Handle connection events
        // 3. Set up message handlers
    }

    public void SubscribeToTopic(string topic, System.Action<string> callback)
    {
        // Subscribe to ROS topic via rosbridge
        // This would send a subscribe message via WebSocket
        Debug.Log($"Subscribing to topic: {topic}");
    }

    public void PublishToTopic(string topic, string message)
    {
        // Publish message to ROS topic via rosbridge
        // This would send a publish message via WebSocket
        Debug.Log($"Publishing to topic: {topic}");
    }

    // Example: Send velocity command
    public void SendVelocityCommand(float linear, float angular)
    {
        string cmd = $"{{\"linear\": {{\"x\": {linear}, \"y\": 0, \"z\": 0}}, " +
                     $"\"angular\": {{\"x\": 0, \"y\": 0, \"z\": {angular}}}}}";

        PublishToTopic($"{robotNamespace}/cmd_vel", cmd);
    }

    // Example: Subscribe to laser scan
    public void SubscribeToLaserScan(System.Action<string> callback)
    {
        SubscribeToTopic($"{robotNamespace}/scan", callback);
    }
}
```

### Alternative: ROS# Unity Package

ROS# is a Unity package that provides ROS integration:

1. **Import ROS#** into your Unity project
2. **Configure ROS Master IP** in the ROSConnection component
3. **Use provided components** for common ROS operations

```csharp
// Using ROS# (if available)
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

public class RosIntegratedRobotController : MonoBehaviour
{
    private ROSConnection ros;
    private string robotNamespace = "/my_robot";

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>($"{robotNamespace}/cmd_vel");

        // Subscribe to sensor topics
        ros.Subscribe<LaserScanMsg>($"{robotNamespace}/scan", OnLaserScan);
    }

    public void MoveRobot(float linear, float angular)
    {
        var twist = new TwistMsg();
        twist.linear = new Vector3Msg(linear, 0, 0);
        twist.angular = new Vector3Msg(0, 0, angular);

        ros.Publish($"{robotNamespace}/cmd_vel", twist);
    }

    void OnLaserScan(LaserScanMsg scan)
    {
        // Process laser scan data
        Debug.Log($"Received scan with {scan.ranges.Length} points");
        // Update visualization based on scan data
    }
}
```

## Monitoring and Visualization

### Real-time Data Visualization

```csharp
// RealTimeDataVisualizer.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class RealTimeDataVisualizer : MonoBehaviour
{
    [Header("Data Visualization")]
    public LineRenderer dataLineRenderer;
    public int maxDataPoints = 100;
    public float timeWindow = 10f;  // seconds

    [Header("Sensor Data")]
    public Text sensorValueText;
    public Slider sensorValueSlider;

    private Queue<float> dataPoints;
    private Queue<float> timeStamps;

    void Start()
    {
        InitializeVisualization();
    }

    void InitializeVisualization()
    {
        dataPoints = new Queue<float>();
        timeStamps = new Queue<float>();

        if (dataLineRenderer != null)
        {
            dataLineRenderer.positionCount = 0;
        }
    }

    public void AddDataPoint(float value)
    {
        float currentTime = Time.time;

        // Add new data point
        dataPoints.Enqueue(value);
        timeStamps.Enqueue(currentTime);

        // Remove old data points outside the time window
        while (timeStamps.Count > 0 &&
               (currentTime - timeStamps.Peek()) > timeWindow)
        {
            dataPoints.Dequeue();
            timeStamps.Dequeue();
        }

        // Update visualization
        UpdateVisualization();

        // Update UI elements
        if (sensorValueText != null)
        {
            sensorValueText.text = $"Value: {value:F3}";
        }

        if (sensorValueSlider != null)
        {
            // Normalize value for slider (assuming 0-10 range for example)
            sensorValueSlider.value = Mathf.Clamp01(value / 10f);
        }
    }

    void UpdateVisualization()
    {
        if (dataLineRenderer == null || dataPoints.Count < 2) return;

        // Convert queue to arrays for processing
        float[] values = new float[dataPoints.Count];
        float[] times = new float[timeStamps.Count];

        dataPoints.CopyTo(values, 0);
        timeStamps.CopyTo(times, 0);

        // Calculate positions for line renderer
        Vector3[] positions = new Vector3[values.Length];
        float timeStart = times[0];
        float timeEnd = times[times.Length - 1];
        float timeRange = timeEnd - timeStart;

        for (int i = 0; i < values.Length; i++)
        {
            float timeNormalized = (times[i] - timeStart) / timeRange;
            float valueNormalized = Mathf.InverseLerp(-10f, 10f, values[i]); // Assuming -10 to 10 range

            positions[i] = new Vector3(
                timeNormalized * 10f - 5f,  // X: time (scaled)
                valueNormalized * 5f - 2.5f, // Y: value (scaled)
                0f
            );
        }

        dataLineRenderer.positionCount = positions.Length;
        dataLineRenderer.SetPositions(positions);
    }

    public void ClearData()
    {
        dataPoints.Clear();
        timeStamps.Clear();

        if (dataLineRenderer != null)
        {
            dataLineRenderer.positionCount = 0;
        }
    }
}
```

### 3D Sensor Data Visualization

```csharp
// SensorDataVisualizer3D.cs
using UnityEngine;
using System.Collections.Generic;

public class SensorDataVisualizer3D : MonoBehaviour
{
    [Header("LiDAR Visualization")]
    public GameObject lidarPointPrefab;
    public Color lidarColor = Color.red;
    public float pointScale = 0.05f;

    [Header("Camera Feed")]
    public RawImage cameraFeedImage;
    public Renderer cameraTextureRenderer;

    private List<GameObject> lidarPoints;

    void Start()
    {
        lidarPoints = new List<GameObject>();
    }

    public void UpdateLidarVisualization(float[] ranges, float[] angles, Vector3 robotPosition)
    {
        // Clear previous points
        ClearLidarPoints();

        // Create new points based on ranges and angles
        for (int i = 0; i < ranges.Length; i++)
        {
            if (ranges[i] > 0 && ranges[i] < 30) // Valid range
            {
                float angle = angles[i];
                Vector3 direction = new Vector3(
                    Mathf.Cos(angle) * ranges[i],
                    0,
                    Mathf.Sin(angle) * ranges[i]
                );

                Vector3 worldPosition = robotPosition + direction;

                GameObject point = Instantiate(lidarPointPrefab, worldPosition, Quaternion.identity);
                point.transform.localScale = Vector3.one * pointScale;

                // Set color
                Renderer pointRenderer = point.GetComponent<Renderer>();
                if (pointRenderer != null)
                {
                    pointRenderer.material.color = lidarColor;
                }

                lidarPoints.Add(point);
            }
        }
    }

    public void UpdateCameraFeed(Texture2D image)
    {
        if (cameraFeedImage != null)
        {
            cameraFeedImage.texture = image;
        }

        if (cameraTextureRenderer != null)
        {
            cameraTextureRenderer.material.mainTexture = image;
        }
    }

    void ClearLidarPoints()
    {
        foreach (GameObject point in lidarPoints)
        {
            if (point != null)
            {
                DestroyImmediate(point);
            }
        }
        lidarPoints.Clear();
    }
}
```

## User Experience Considerations

### Designing for Different User Types

#### Expert Operators
- **Detailed information** and advanced controls
- **Efficiency** over simplicity
- **Customizable interfaces**
- **Keyboard shortcuts** and quick access

#### Novice Users
- **Simple, intuitive** interfaces
- **Guided workflows**
- **Clear feedback** and instructions
- **Limited options** to avoid confusion

#### Emergency Responders
- **Critical information** prioritized
- **Quick access** to emergency functions
- **Clear status** indicators
- **Redundant controls** for safety

### Accessibility Considerations

```csharp
// AccessibilityManager.cs
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class AccessibilityManager : MonoBehaviour
{
    [Header("Visual Accessibility")]
    public float minimumContrastRatio = 4.5f;
    public bool highContrastMode = false;
    public float fontSizeMultiplier = 1.0f;

    [Header("Control Accessibility")]
    public bool keyboardNavigation = true;
    public bool gamepadSupport = true;
    public float keyboardRepeatDelay = 0.5f;
    public float keyboardRepeatRate = 0.05f;

    [Header("Audio Feedback")]
    public bool audioEnabled = true;
    public AudioSource audioSource;

    void Start()
    {
        ApplyAccessibilitySettings();
    }

    public void ApplyAccessibilitySettings()
    {
        // Apply high contrast mode
        if (highContrastMode)
        {
            EnableHighContrastMode();
        }

        // Adjust font sizes
        AdjustFontSizes(fontSizeMultiplier);

        // Setup keyboard navigation
        if (keyboardNavigation)
        {
            SetupKeyboardNavigation();
        }
    }

    void EnableHighContrastMode()
    {
        // Change colors to high contrast scheme
        // This would iterate through UI elements and adjust their colors
        Graphic[] graphics = FindObjectsOfType<Graphic>();
        foreach (Graphic graphic in graphics)
        {
            // Apply high contrast colors
            if (graphic is Text text)
            {
                text.color = Color.black;
                // Set background to white or vice versa
            }
        }
    }

    void AdjustFontSizes(float multiplier)
    {
        Text[] texts = FindObjectsOfType<Text>();
        foreach (Text text in texts)
        {
            text.fontSize = Mathf.RoundToInt(text.fontSize * multiplier);
        }
    }

    void SetupKeyboardNavigation()
    {
        Selectable[] selectables = FindObjectsOfType<Selectable>();
        for (int i = 0; i < selectables.Length - 1; i++)
        {
            Selectable current = selectables[i];
            Selectable next = selectables[i + 1];

            // Set navigation to move to next element
            Navigation nav = current.navigation;
            nav.mode = Navigation.Mode.Explicit;
            nav.selectOnDown = next;
            nav.selectOnRight = next;
            current.navigation = nav;
        }
    }

    public void PlayAudioFeedback(string eventDescription)
    {
        if (audioEnabled && audioSource != null)
        {
            // Play appropriate audio feedback based on event
            // Implementation would depend on your audio system
            Debug.Log($"Audio feedback: {eventDescription}");
        }
    }
}
```

### Performance Optimization

```csharp
// PerformanceOptimizer.cs
using UnityEngine;
using System.Collections.Generic;

public class PerformanceOptimizer : MonoBehaviour
{
    [Header("LOD Settings")]
    public int maxVisibleRobots = 10;
    public float lodDistance = 50f;

    [Header("Visualization Settings")]
    public bool enableDetailedVisualization = true;
    public bool enableShadows = true;
    public int maxLidarPoints = 1000;

    [Header("Quality Settings")]
    public int targetFrameRate = 60;

    private List<Renderer> robotRenderers;
    private int currentRobotCount = 0;

    void Start()
    {
        Application.targetFrameRate = targetFrameRate;
        robotRenderers = new List<Renderer>();
    }

    void Update()
    {
        OptimizePerformance();
    }

    void OptimizePerformance()
    {
        // Adjust level of detail based on distance
        AdjustLOD();

        // Limit detailed visualization
        if (!enableDetailedVisualization)
        {
            SimplifyVisualizations();
        }

        // Limit number of visible robots
        if (currentRobotCount > maxVisibleRobots)
        {
            CullDistantRobots();
        }
    }

    void AdjustLOD()
    {
        // This would adjust the level of detail for distant objects
        foreach (Renderer renderer in robotRenderers)
        {
            float distance = Vector3.Distance(renderer.transform.position, Camera.main.transform.position);

            if (distance > lodDistance)
            {
                // Use simpler mesh or hide if too far
                renderer.enabled = false;
            }
            else
            {
                renderer.enabled = true;
            }
        }
    }

    void SimplifyVisualizations()
    {
        // Reduce complexity of visualizations
        // For example, reduce number of LiDAR points displayed
    }

    void CullDistantRobots()
    {
        // Hide robots that are too far away
    }
}
```

## Practical Example: Complete HRI Interface

Here's a complete example that combines all the concepts:

```csharp
// CompleteHRIInterface.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class CompleteHRIInterface : MonoBehaviour
{
    [Header("UI Panels")]
    public GameObject mainPanel;
    public GameObject teleopPanel;
    public GameObject monitoringPanel;
    public GameObject settingsPanel;

    [Header("Robot Control")]
    public RobotModelController robotModel;
    public TeleoperationController teleopController;
    public DashboardManager dashboard;

    [Header("Sensors")]
    public RealTimeDataVisualizer dataVisualizer;
    public SensorDataVisualizer3D sensorVisualizer;

    [Header("Emergency")]
    public Button emergencyStopButton;

    private Dictionary<string, GameObject> panels;
    private string currentPanel = "main";

    void Start()
    {
        InitializeInterface();
        SetupEmergencySystems();
    }

    void InitializeInterface()
    {
        panels = new Dictionary<string, GameObject>
        {
            {"main", mainPanel},
            {"teleop", teleopPanel},
            {"monitoring", monitoringPanel},
            {"settings", settingsPanel}
        };

        // Hide all panels except main
        foreach (var panel in panels.Values)
        {
            if (panel != mainPanel)
            {
                panel.SetActive(false);
            }
        }

        // Initialize all components
        if (robotModel != null) robotModel.enabled = true;
        if (teleopController != null) teleopController.enabled = true;
        if (dashboard != null) dashboard.enabled = true;
        if (dataVisualizer != null) dataVisualizer.enabled = true;
        if (sensorVisualizer != null) sensorVisualizer.enabled = true;
    }

    void SetupEmergencySystems()
    {
        if (emergencyStopButton != null)
        {
            emergencyStopButton.onClick.AddListener(EmergencyStop);
        }
    }

    public void SwitchToPanel(string panelName)
    {
        // Hide current panel
        if (panels.ContainsKey(currentPanel))
        {
            panels[currentPanel].SetActive(false);
        }

        // Show new panel
        if (panels.ContainsKey(panelName))
        {
            panels[panelName].SetActive(true);
            currentPanel = panelName;
        }
    }

    void EmergencyStop()
    {
        Debug.LogError("EMERGENCY STOP - ALL SYSTEMS HALTED!");

        // Stop all robot movement
        if (teleopController != null)
        {
            teleopController.StopRobot();
        }

        // Clear all visualizations
        if (dataVisualizer != null)
        {
            dataVisualizer.ClearData();
        }

        // Send emergency stop to robot (in real implementation)
        SendEmergencyStopToRobot();
    }

    void SendEmergencyStopToRobot()
    {
        // In real implementation, send emergency stop command via ROS
        Debug.Log("Emergency stop command sent to robot");
    }

    // Methods to update from external data sources
    public void UpdateRobotPose(Vector3 position, Vector3 rotation)
    {
        if (robotModel != null)
        {
            robotModel.transform.position = position;
            robotModel.transform.eulerAngles = rotation;
        }
    }

    public void UpdateSensorData(float[] lidarRanges, float[] lidarAngles)
    {
        if (sensorVisualizer != null)
        {
            sensorVisualizer.UpdateLidarVisualization(lidarRanges, lidarAngles,
                robotModel != null ? robotModel.transform.position : Vector3.zero);
        }
    }

    public void UpdateSystemStatus(SystemStatus status)
    {
        if (dashboard != null)
        {
            // Dashboard will update automatically in its Update method
        }
    }
}
```

## Summary and Next Steps

In this chapter, you learned:
- How to use Unity for creating human-robot interaction interfaces
- How to visualize robots and sensor data in 3D environments
- How to design intuitive teleoperation interfaces
- How to create monitoring dashboards for robot status
- How to consider user experience in HRI design
- How to optimize performance for real-time applications

### Key Takeaways

- Unity provides powerful tools for creating engaging HRI interfaces
- Good HRI design prioritizes safety, usability, and feedback
- Performance optimization is crucial for real-time applications
- Accessibility should be considered in all HRI interfaces
- Integration with ROS enables real-time data exchange

### Next Steps

In the next module, you'll learn about NVIDIA Isaac for AI perception and navigation, where you'll combine these HRI interfaces with advanced AI capabilities.

## Exercises

1. Create a Unity scene with a simple robot model and basic teleoperation controls
2. Implement a monitoring dashboard showing robot status and sensor data
3. Design an accessible interface for users with different abilities
4. Create a VR interface for immersive robot teleoperation
5. Integrate your Unity interface with ROS using rosbridge

## References

- Unity Documentation: https://docs.unity3d.com/
- ROS Integration: https://github.com/Unity-Technologies/ROS-TCP-Connector
- Human-Robot Interaction Research: https://www.hri2023.org/
- Unity UI System: https://docs.unity3d.com/Manual/UI.html
- VR/AR Development in Unity: https://docs.unity3d.com/Manual/VROverview.html