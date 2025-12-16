// Auto-generated quiz module
// This file is generated during the build process from quiz markdown files

export const module1_quiz = {
  "id": "module1",
  "title": "Module 1 Quiz - ROS 2 Fundamentals",
  "questions": [
    {
      "id": "q1",
      "text": "What does ROS 2 stand for?",
      "options": [
        "Robot Operating System 2",
        "Robotic Operations Suite 2",
        "Robot Operating Software 2",
        "Robotic Operating System 2"
      ],
      "correctAnswer": 0,
      "explanation": "Explanation for question 1"
    },
    {
      "id": "q2",
      "text": "Which communication pattern uses a publish/subscribe model in ROS 2?",
      "options": [
        "Services",
        "Actions",
        "Topics",
        "Parameters"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 2"
    },
    {
      "id": "q3",
      "text": "What is the primary programming language used for the ROS 2 client library rclpy?",
      "options": [
        "C++",
        "Python",
        "Java",
        "JavaScript"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 3"
    },
    {
      "id": "q4",
      "text": "In URDF, what does the 'link' element represent?",
      "options": [
        "A connection between two parts",
        "A rigid part of the robot",
        "A sensor on the robot",
        "A joint in the robot"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 4"
    },
    {
      "id": "q5",
      "text": "Which of the following is NOT a valid joint type in URDF?",
      "options": [
        "revolute",
        "continuous",
        "prismatic",
        "oscillating"
      ],
      "correctAnswer": 3,
      "explanation": "Explanation for question 5"
    },
    {
      "id": "q6",
      "text": "What does DDS stand for in the context of ROS 2?",
      "options": [
        "Distributed Data System",
        "Data Distribution Service",
        "Dynamic Data Sharing",
        "Distributed Development System"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 6"
    },
    {
      "id": "q7",
      "text": "In rclpy, which method is used to create a publisher?",
      "options": [
        "create_publisher()",
        "make_publisher()",
        "init_publisher()",
        "new_publisher()"
      ],
      "correctAnswer": 0,
      "explanation": "Explanation for question 7"
    },
    {
      "id": "q8",
      "text": "What is the purpose of the robot_state_publisher node?",
      "options": [
        "To publish sensor data from the robot",
        "To publish joint states to the TF system",
        "To control robot movement",
        "To manage robot parameters"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 8"
    },
    {
      "id": "q9",
      "text": "Which command is used to run a ROS 2 node?",
      "options": [
        "ros2 run <package_name> <node_name>",
        "ros2 execute <package_name> <node_name>",
        "ros2 start <package_name> <node_name>",
        "ros2 launch <package_name> <node_name>"
      ],
      "correctAnswer": 0,
      "explanation": "Explanation for question 9"
    },
    {
      "id": "q10",
      "text": "In a URDF file, what does the 'inertial' element define?",
      "options": [
        "How the link looks visually",
        "How the link collides with other objects",
        "The physical properties for physics simulation",
        "The joint properties of the link"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 10"
    },
    {
      "id": "q11",
      "text": "What is the default middleware used by ROS 2 for communication?",
      "options": [
        "TCP/IP",
        "ZeroMQ",
        "DDS (Data Distribution Service)",
        "MQTT"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 11"
    },
    {
      "id": "q12",
      "text": "Which rclpy function is used to initialize the ROS 2 client library?",
      "options": [
        "rclpy.start()",
        "rclpy.init()",
        "rclpy.begin()",
        "rclpy.setup()"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 12"
    }
  ]
};

export const module2_quiz = {
  "id": "module2",
  "title": "Module 2 Quiz - Simulation Environments",
  "questions": [
    {
      "id": "q1",
      "text": "What does Gazebo use as its default physics engine?",
      "options": [
        "Bullet",
        "ODE (Open Dynamics Engine)",
        "PhysX",
        "Havok"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 1"
    },
    {
      "id": "q2",
      "text": "In SDF (Simulation Description Format), what element defines the physics properties of a world?",
      "options": [
        "`<physics>`",
        "`<engine>`",
        "`<simulation>`",
        "`<properties>`"
      ],
      "correctAnswer": 0,
      "explanation": "Explanation for question 2"
    },
    {
      "id": "q3",
      "text": "Which of the following is NOT a valid joint type in URDF?",
      "options": [
        "revolute",
        "prismatic",
        "continuous",
        "oscillating"
      ],
      "correctAnswer": 3,
      "explanation": "Explanation for question 3"
    },
    {
      "id": "q4",
      "text": "What is the primary purpose of the robot_state_publisher in ROS 2?",
      "options": [
        "To publish sensor data from the robot",
        "To publish joint states to the TF system",
        "To control robot movement",
        "To manage robot parameters"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 4"
    },
    {
      "id": "q5",
      "text": "In Gazebo, what sensor type would you use to simulate a LiDAR?",
      "options": [
        "camera",
        "depth",
        "ray",
        "gpu_ray"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 5"
    },
    {
      "id": "q6",
      "text": "What is the coordinate system convention used by ROS?",
      "options": [
        "X-right, Y-up, Z-forward (left-handed)",
        "X-forward, Y-left, Z-up (right-handed)",
        "X-right, Y-forward, Z-up (left-handed)",
        "X-forward, Y-right, Z-down (right-handed)"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 6"
    },
    {
      "id": "q7",
      "text": "Which Unity coordinate system axis represents \"up\" in the default setup?",
      "options": [
        "X-axis",
        "Y-axis",
        "Z-axis",
        "None of the above"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 7"
    },
    {
      "id": "q8",
      "text": "What does the \"DDS\" in ROS 2 stand for?",
      "options": [
        "Distributed Data System",
        "Data Distribution Service",
        "Dynamic Data Sharing",
        "Distributed Development System"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 8"
    },
    {
      "id": "q9",
      "text": "In a Gazebo world file, which element defines a light source?",
      "options": [
        "`<lighting>`",
        "`<source>`",
        "`<light>`",
        "`<illumination>`"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 9"
    },
    {
      "id": "q10",
      "text": "What is the purpose of the URDF \"inertial\" element?",
      "options": [
        "To define how the link looks visually",
        "To define collision properties",
        "To define physical properties for physics simulation",
        "To define joint properties"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 10"
    },
    {
      "id": "q11",
      "text": "Which ROS 2 message type is typically used for LiDAR data?",
      "options": [
        "sensor_msgs/Image",
        "sensor_msgs/LaserScan",
        "sensor_msgs/PointCloud2",
        "geometry_msgs/Point"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 11"
    },
    {
      "id": "q12",
      "text": "In Unity, what system is used to create user interfaces?",
      "options": [
        "UI System",
        "Canvas System",
        "Both A and B",
        "Graphics System"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 12"
    },
    {
      "id": "q13",
      "text": "What is the main advantage of using simulation in robotics development?",
      "options": [
        "It's always faster than real hardware",
        "It allows safe testing without risk to hardware",
        "It perfectly replicates real-world conditions",
        "It requires no computational resources"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 13"
    },
    {
      "id": "q14",
      "text": "Which of these is a common sensor configuration parameter for noise modeling?",
      "options": [
        "Standard deviation",
        "Bias",
        "Drift",
        "All of the above"
      ],
      "correctAnswer": 3,
      "explanation": "Explanation for question 14"
    },
    {
      "id": "q15",
      "text": "In a behavior tree for robot control, what does a \"sequence\" node do?",
      "options": [
        "Executes all children in parallel",
        "Executes children until one fails",
        "Executes children until one succeeds",
        "Randomly selects a child to execute"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 15"
    },
    {
      "id": "q16",
      "text": "What is the primary purpose of the IMU sensor in robotics?",
      "options": [
        "To capture visual information",
        "To measure position and orientation",
        "To measure angular velocity and linear acceleration",
        "To detect obstacles"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 16"
    },
    {
      "id": "q17",
      "text": "In Gazebo, what does the \"update_rate\" parameter specify for a sensor?",
      "options": [
        "The maximum range of the sensor",
        "How often the sensor publishes data (Hz)",
        "The resolution of the sensor",
        "The accuracy of the sensor"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 17"
    },
    {
      "id": "q18",
      "text": "Which Unity component is essential for creating interactive 3D objects?",
      "options": [
        "Rigidbody",
        "Collider",
        "Both A and B",
        "MeshRenderer"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 18"
    },
    {
      "id": "q19",
      "text": "What does the \"URDF\" in robotics stand for?",
      "options": [
        "Universal Robot Data Format",
        "Unified Robot Description Format",
        "Universal Robot Description Format",
        "Unified Robot Data Format"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 19"
    },
    {
      "id": "q20",
      "text": "In ROS 2, what command is used to visualize the TF tree?",
      "options": [
        "ros2 run tf2_tools view_frames",
        "ros2 run rqt_tf_tree tf_tree",
        "ros2 run rviz2 rviz2",
        "Both A and B"
      ],
      "correctAnswer": 3,
      "explanation": "Explanation for question 20"
    }
  ]
};

export const module3_quiz = {
  "id": "module3",
  "title": "Module 3 Quiz: Isaac Sim, Isaac ROS, VSLAM, and Navigation for Humanoids",
  "questions": [
    {
      "id": "q1",
      "text": "What is the primary purpose of Isaac Sim in robotics development?",
      "options": [
        "Real-time control of physical robots",
        "High-fidelity physics simulation and photorealistic rendering for robotics",
        "Speech recognition for human-robot interaction",
        "Low-level motor control"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 1"
    },
    {
      "id": "q2",
      "text": "Which physics engine does Isaac Sim primarily use for simulation?",
      "options": [
        "Bullet Physics",
        "PhysX",
        "Havok",
        "ODE (Open Dynamics Engine)"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 2"
    },
    {
      "id": "q3",
      "text": "What does VSLAM stand for in the context of robotics?",
      "options": [
        "Visual Simultaneous Localization and Mapping",
        "Virtual Sensor Localization and Mapping",
        "Vision-based System for Localization and Mapping",
        "Variable Sensor Localization and Mapping"
      ],
      "correctAnswer": 0,
      "explanation": "Explanation for question 3"
    },
    {
      "id": "q4",
      "text": "In Isaac ROS, which sensor processing acceleration is specifically designed for stereo vision?",
      "options": [
        "Stereo DNN",
        "Stereo Image Proc",
        "Stereo Disparity",
        "Stereo Visual Odometry"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 4"
    },
    {
      "id": "q5",
      "text": "What is the main advantage of using synthetic data generation in Isaac Sim?",
      "options": [
        "Lower computational requirements",
        "Perfectly labeled training data for AI models",
        "Direct transfer to real-world applications without domain adaptation",
        "Real-time performance in all scenarios"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 5"
    },
    {
      "id": "q6",
      "text": "Which coordinate system does Isaac Sim use by default?",
      "options": [
        "Right-handed system (X-forward, Y-left, Z-up)",
        "Left-handed system (X-right, Y-up, Z-forward)",
        "Right-handed system (X-right, Y-forward, Z-up)",
        "Left-handed system (X-right, Y-up, Z-forward)"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 6"
    },
    {
      "id": "q7",
      "text": "What is the Zero Moment Point (ZMP) in humanoid robotics?",
      "options": [
        "in humanoid robotics?",
        "The point where the robot's center of mass is located",
        "The point where the sum of all moments caused by gravity and inertia forces equals zero",
        "The point where the robot's feet touch the ground",
        "The point where the robot's vision system is focused"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 7"
    },
    {
      "id": "q8",
      "text": "Which of the following is NOT a key feature of Isaac ROS?",
      "options": [
        "GPU-accelerated computer vision",
        "Hardware abstraction for sensors",
        "Built-in path planning algorithms",
        "Real-time perception pipelines"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 8"
    },
    {
      "id": "q9",
      "text": "In Nav2 for humanoids, what is the primary challenge that differentiates it from wheeled robot navigation?",
      "options": [
        "Higher computational requirements",
        "The need to maintain dynamic balance during locomotion",
        "More complex sensor configurations",
        "Larger environmental maps"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 9"
    },
    {
      "id": "q10",
      "text": "What is the main purpose of the Behavior Tree in Nav2?",
      "options": [
        "To store environmental maps",
        "To define the decision-making logic for navigation",
        "To manage robot's joint positions",
        "To store sensor calibration data"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 10"
    }
  ]
};

export const module4_quiz = {
  "id": "module4",
  "title": "Module 4 Quiz: Voice-to-Action and LLM Cognitive Planning",
  "questions": [
    {
      "id": "q1",
      "text": "What is the primary function of OpenAI's Whisper model in robotics applications?",
      "options": [
        "Motion planning for robot arms",
        "Speech recognition and transcription",
        "Computer vision processing",
        "Path planning for navigation"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 1"
    },
    {
      "id": "q2",
      "text": "Which of the following is NOT a common Whisper model size?",
      "options": [
        "Tiny",
        "Base",
        "Large",
        "Mega"
      ],
      "correctAnswer": 3,
      "explanation": "Explanation for question 2"
    },
    {
      "id": "q3",
      "text": "In LLM-based cognitive planning for robots, what does \"zero-shot learning\" refer to?",
      "options": [
        "Learning without any prior training",
        "Performing tasks without examples in the prompt",
        "Running the model without any parameters",
        "Operating without any sensors"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 3"
    },
    {
      "id": "q4",
      "text": "What is the main challenge of integrating LLMs with real-time robotic systems?",
      "options": [
        "High computational requirements",
        "Latency constraints for real-time response",
        "Limited vocabulary of LLMs",
        "Inability to process sensor data"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 4"
    },
    {
      "id": "q5",
      "text": "Which safety consideration is most critical when using LLMs for robot control?",
      "options": [
        "Model accuracy",
        "Response time",
        "Potential for unsafe action generation",
        "Data privacy"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 5"
    },
    {
      "id": "q6",
      "text": "What is the typical sampling rate for audio processing in voice-controlled robotics?",
      "options": [
        "8000 Hz",
        "11025 Hz",
        "16000 Hz",
        "44100 Hz"
      ],
      "correctAnswer": 2,
      "explanation": "Explanation for question 6"
    },
    {
      "id": "q7",
      "text": "In cognitive robotics, what is the purpose of a \"belief state\"?",
      "options": [
        "The robot's confidence in its sensors",
        "A representation of the robot's knowledge about the world",
        "The robot's emotional state",
        "The robot's battery level"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 7"
    },
    {
      "id": "q8",
      "text": "Which of the following is a key component of uncertainty-aware planning?",
      "options": [
        "Deterministic action sequences",
        "Probabilistic reasoning about outcomes",
        "Fixed execution timelines",
        "Open-loop control systems"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 8"
    },
    {
      "id": "q9",
      "text": "What is the main advantage of using transformer-based models like Whisper for speech recognition in robotics?",
      "options": [
        "Lower computational requirements",
        "Better handling of context and long-term dependencies",
        "Simpler implementation",
        "Real-time processing capabilities"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 9"
    },
    {
      "id": "q10",
      "text": "In voice command processing, what is the purpose of a wake word detection system?",
      "options": [
        "To improve speech recognition accuracy",
        "To activate the voice processing system when needed",
        "To translate speech to text",
        "To filter out background noise"
      ],
      "correctAnswer": 1,
      "explanation": "Explanation for question 10"
    }
  ]
};

// Export all quizzes as a single object
export const allQuizzes = {
  "module1": module1_quiz,
  "module2": module2_quiz,
  "module3": module3_quiz,
  "module4": module4_quiz
};
