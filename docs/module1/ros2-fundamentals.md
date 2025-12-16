---
sidebar_position: 2
---

# ROS 2 Fundamentals

## Overview

This chapter introduces you to the core concepts of ROS 2 (Robot Operating System 2), the middleware that serves as the nervous system for robotic applications. ROS 2 is a flexible framework for writing robot software, providing services such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the core concepts of ROS 2 architecture
- Identify the key components of ROS 2 (nodes, topics, services, actions)
- Understand the communication patterns in ROS 2
- Set up a basic ROS 2 environment
- Create and run simple ROS 2 examples

## Table of Contents

1. [Introduction to ROS 2](#introduction-to-ros-2)
2. [ROS 2 Architecture](#ros-2-architecture)
3. [Nodes](#nodes)
4. [Topics and Message Passing](#topics-and-message-passing)
5. [Services](#services)
6. [Actions](#actions)
7. [Workspaces and Packages](#workspaces-and-packages)
8. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is not an operating system but rather a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Features of ROS 2

- **Distributed computing**: Nodes can run on different machines and communicate over a network
- **Language independence**: Supports multiple programming languages (C++, Python, etc.)
- **Hardware abstraction**: Provides interfaces to various sensors and actuators
- **Modular design**: Components can be developed and tested independently
- **Large community**: Extensive ecosystem of packages and tools

### Why ROS 2?

ROS 2 is the next generation of the Robot Operating System, designed to address limitations of the original ROS and provide enhanced capabilities:

- Improved security and safety features
- Better real-time support
- Enhanced scalability
- More robust communication middleware (DDS - Data Distribution Service)
- Better support for commercial applications

## ROS 2 Architecture

![ROS 2 Architecture](../../assets/diagrams/ros2_architecture.svg)

### The ROS 2 Graph

The ROS 2 graph represents the network of nodes and their connections. Key concepts include:

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous request/goal-based communication
- **Parameters**: Configuration values shared across nodes

### Communication Middleware

ROS 2 uses DDS (Data Distribution Service) as its communication middleware. DDS provides:

- Publish/subscribe communication patterns
- Service-based communication
- Quality of Service (QoS) policies
- Discovery and matching mechanisms

## Nodes

Nodes are the fundamental building blocks of ROS 2. A node is a process that performs computation. In ROS 2, nodes are designed to be modular and focused on a single task.

### Creating a Node

In Python, a node is created by subclassing `rclpy.Node`:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    minimal_node = MinimalNode()

    try:
        rclpy.spin(minimal_node)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

Nodes in ROS 2 have a well-defined lifecycle:

1. **Unconfigured**: Node is created but not yet configured
2. **Inactive**: Node is configured but not active
3. **Active**: Node is running and performing its tasks
4. **Finalized**: Node is shutting down

## Topics and Message Passing

Topics are the primary method of communication in ROS 2. They use a publish/subscribe pattern where nodes publish messages to topics and other nodes subscribe to topics to receive messages.

### Publishers

A publisher sends messages to a topic:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Subscribers

A subscriber receives messages from a topic:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

### Quality of Service (QoS)

QoS policies allow you to specify requirements for communication:

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local durability
- **History**: Keep last N messages or keep all messages
- **Deadline**: Maximum time between consecutive messages

## Services

Services provide synchronous request/response communication between nodes. A service client sends a request to a service server, which processes the request and returns a response.

### Creating a Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Creating a Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions

Actions are used for long-running tasks that provide feedback during execution. They support goal preemption and provide status updates.

### Action Structure

An action has three message types:
- **Goal**: Requested action
- **Result**: Final outcome
- **Feedback**: Progress updates during execution

## Workspaces and Packages

ROS 2 organizes code into packages, which are grouped into workspaces.

### Creating a Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Package Structure

A typical ROS 2 package contains:
- `package.xml`: Package manifest
- `CMakeLists.txt`: Build configuration for C++
- `setup.py`: Build configuration for Python
- `src/`: Source code
- `test/`: Test files
- `launch/`: Launch files
- `config/`: Configuration files

## Hands-On Example: Publisher-Subscriber

Let's create a simple publisher-subscriber example:

1. Create a new package:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_tutorials
```

2. Create the publisher code in `my_robot_tutorials/my_robot_tutorials/talker.py`:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. Create the subscriber code in `my_robot_tutorials/my_robot_tutorials/listener.py`:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

4. Run the example:
Terminal 1:
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run my_robot_tutorials talker
```

Terminal 2:
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run my_robot_tutorials listener
```

## Summary and Next Steps

In this chapter, you learned the fundamental concepts of ROS 2:
- The distributed architecture and node-based design
- Communication patterns: topics, services, and actions
- How to create simple publishers and subscribers
- The structure of ROS 2 workspaces and packages

### Key Takeaways

- ROS 2 provides a flexible framework for robot software development
- Nodes communicate through topics (publish/subscribe), services (request/response), and actions (goal-based)
- The middleware (DDS) handles the communication infrastructure
- Quality of Service policies allow fine-tuning communication behavior

### Next Steps

In the next chapter, you'll learn how to create Python agents that interact with ROS controllers using rclpy, building on the concepts learned here.

## Exercises

1. Create a publisher that publishes temperature readings to a topic
2. Create a subscriber that logs received messages to a file
3. Modify the publisher to include a timestamp in each message
4. Research and explain the difference between ROS 1 and ROS 2

## References

- ROS 2 Documentation: https://docs.ros.org/
- ROS 2 Tutorials: https://docs.ros.org/en/rolling/Tutorials.html
- DDS Specification: https://www.omg.org/spec/DDS/