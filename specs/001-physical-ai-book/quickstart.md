# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Prerequisites

Before starting with the Physical AI & Humanoid Robotics textbook, ensure you have the following:

- Basic Python programming knowledge (variables, functions, classes)
- Familiarity with command-line interfaces
- A computer with at least 8GB RAM (16GB recommended for simulation work)
- Administrative access to install software (ROS 2, Gazebo, etc.)

## Setting Up Your Environment

### 1. Install ROS 2 (Humble Hawksbill)

Choose your operating system:

**Ubuntu 22.04:**
```bash
# Add the ROS 2 apt repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

# Add the ROS 2 GPG key and repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions
```

**Windows/Other OS:** Follow the official ROS 2 installation guide for your platform.

### 2. Install Additional Dependencies

```bash
# Install Python dependencies for examples
pip3 install rclpy transforms3d numpy matplotlib

# Install Gazebo (if not included in ROS 2 desktop)
sudo apt install ros-humble-gazebo-ros-pkgs
```

### 3. Clone the Book Resources

```bash
# Create workspace directory
mkdir -p ~/book_ws/src
cd ~/book_ws

# Clone the example code repository (or download the examples from the book)
git clone https://github.com/your-book-repo/examples.git src/examples
```

## Getting Started with the First Module

### Module 1: The Robotic Nervous System (ROS 2)

1. **Navigate to the Module 1 directory:**
   ```bash
   cd ~/book_ws
   source /opt/ros/humble/setup.bash
   colcon build
   source install/setup.bash
   ```

2. **Run your first ROS 2 example:**
   ```bash
   # Terminal 1 - Start the publisher
   ros2 run examples_rclpy_minimal_publisher publisher_member_function
   ```

   ```bash
   # Terminal 2 - Start the subscriber (in a new terminal)
   ros2 run examples_rclpy_minimal_subscriber subscriber_member_function
   ```

3. **You should see messages being published and received between nodes.**

## Docusaurus Setup for Local Viewing

If you want to build and view the book content locally:

1. **Install Node.js (version 18 or higher)**
2. **Navigate to the book directory**
3. **Install dependencies:**
   ```bash
   npm install
   ```
4. **Start the development server:**
   ```bash
   npm start
   ```
5. **Open your browser to http://localhost:3000**

## Troubleshooting Common Issues

- **ROS 2 commands not found**: Make sure you've sourced the setup.bash file
- **Python modules not found**: Install missing packages with `pip3 install <package>`
- **Gazebo won't start**: Check if your graphics drivers support OpenGL
- **Build errors**: Ensure all dependencies are installed and workspace is properly structured

## Next Steps

After completing the setup:
1. Read Module 1 Chapter 1: ROS 2 Fundamentals
2. Try the examples in the chapter
3. Complete the exercises at the end of the chapter
4. Take the Module 1 quiz to assess your understanding

## Getting Help

- Check the official ROS 2 documentation
- Use the book's GitHub repository for issues with examples
- Join the ROS community forums for general questions