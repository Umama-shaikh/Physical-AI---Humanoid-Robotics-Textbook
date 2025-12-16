// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1/index',
        'module1/ros2-fundamentals',
        'module1/python-agents-ros',
        'module1/urdf-humanoids',
        'module1/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2/index',
        'module2/gazebo-simulation',
        'module2/simulated-sensors',
        'module2/unity-hri',
        'module2/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module3/index',
        'module3/isaac-sim-basics',
        'module3/isaac-ros-vslam',
        'module3/nav2-humanoids',
        'module3/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4/index',
        'module4/voice-to-action',
        'module4/llm-planning',
        'module4/capstone-autonomous',
        'module4/quiz',
      ],
    },
  ],
};

module.exports = sidebars;