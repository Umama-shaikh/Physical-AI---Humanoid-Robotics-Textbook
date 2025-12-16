---
sidebar_position: 6
---

# LLM Cognitive Planning: AI-Driven Decision Making for Humanoid Robots

## Overview

This chapter explores the integration of Large Language Models (LLMs) with humanoid robot cognitive architectures for advanced planning and decision-making. You'll learn how to leverage LLMs for high-level task planning, situational awareness, and adaptive behavior generation. The chapter covers both the theoretical foundations and practical implementation of LLM-based cognitive systems for humanoid robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the role of LLMs in robotic cognitive architectures
- Implement LLM-based task planning and reasoning systems
- Integrate LLMs with robot perception and action systems
- Design context-aware planning for dynamic environments
- Optimize LLM usage for real-time robotic applications
- Handle uncertainty and incomplete information in planning

## Table of Contents

1. [Introduction to LLMs in Robotics](#introduction-to-llms-in-robotics)
2. [Cognitive Architecture Overview](#cognitive-architecture-overview)
3. [LLM Integration Patterns](#llm-integration-patterns)
4. [Task Planning with LLMs](#task-planning-with-llms)
5. [Context and Memory Management](#context-and-memory-management)
6. [Perception-Action Integration](#perception-action-integration)
7. [Real-time Considerations](#real-time-considerations)
8. [Uncertainty Handling](#uncertainty-handling)
9. [Safety and Ethics](#safety-and-ethics)
10. [Implementation Examples](#implementation-examples)
11. [Testing and Validation](#testing-and-validation)
12. [Summary and Next Steps](#summary-and-next-steps)

## Introduction to LLMs in Robotics

### The Role of LLMs in Robotic Intelligence

Large Language Models have emerged as powerful tools for robotic cognitive systems, providing:

1. **Natural Language Understanding**: Processing human commands and instructions
2. **Common-Sense Reasoning**: Applying general knowledge to specific situations
3. **Task Decomposition**: Breaking complex goals into executable steps
4. **Contextual Adaptation**: Adjusting behavior based on environment and history
5. **Learning from Interaction**: Improving performance through experience

### Benefits of LLM Integration

- **Flexible Command Interpretation**: Understanding varied ways to express the same intent
- **Knowledge Integration**: Accessing vast amounts of world knowledge
- **Adaptive Planning**: Adjusting plans based on changing circumstances
- **Human-like Interaction**: More natural and intuitive interfaces
- **Generalization**: Applying learned patterns to novel situations

### Challenges in LLM-Robot Integration

- **Real-time Constraints**: LLM inference can be slow for time-critical applications
- **Hallucination**: LLMs may generate incorrect or fabricated information
- **Context Limitations**: Limited memory and context window sizes
- **Safety Concerns**: Potential for unsafe actions based on LLM suggestions
- **Resource Requirements**: High computational and memory demands

## Cognitive Architecture Overview

### Multi-Level Cognitive Architecture

```
High-Level Cognition (LLM) → Task Planning → Action Selection → Motor Control
```

### Component Layers

1. **Symbolic Reasoning Layer**: LLM-based high-level reasoning
2. **Task Planning Layer**: Decomposing goals into subtasks
3. **Behavior Selection Layer**: Choosing appropriate behaviors
4. **Action Execution Layer**: Low-level motor control
5. **Perception Layer**: Environmental state monitoring
6. **Memory Layer**: Context and experience storage

### Integration Points

The cognitive architecture integrates with:
- ROS 2 communication framework
- Robot operating system services
- Perception and sensing systems
- Navigation and manipulation modules
- Human interaction interfaces

## LLM Integration Patterns

### Direct Integration Pattern

```python
# llm_integration.py
import openai
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    SOCIAL = "social"
    MAINTENANCE = "maintenance"

@dataclass
class RobotState:
    """Represents the current state of the robot"""
    position: Dict[str, float]
    battery_level: float
    available_actions: List[str]
    current_task: Optional[str]
    detected_objects: List[Dict[str, Any]]
    human_interactions: List[Dict[str, Any]]

@dataclass
class PlanningRequest:
    """Request for LLM-based planning"""
    goal: str
    current_state: RobotState
    context: Dict[str, Any]
    constraints: List[str]

@dataclass
class PlanningResponse:
    """Response from LLM-based planning"""
    plan: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    estimated_time: float

class LLMPlanner:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

        # System prompt for robotic planning
        self.system_prompt = """
        You are an AI assistant for a humanoid robot. Your role is to help the robot plan actions to achieve goals.
        Consider the robot's capabilities, current state, and environment when creating plans.
        Respond with a structured JSON plan containing: {plan: [{action: string, parameters: object}], reasoning: string, confidence: float, estimated_time: float}
        """

    async def plan_task(self, request: PlanningRequest) -> Optional[PlanningResponse]:
        """Generate a plan for the given request using LLM"""
        try:
            # Construct the user message
            user_message = self._construct_user_message(request)

            # Call the LLM
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Parse the response
            content = response.choices[0].message.content
            return self._parse_response(content)

        except Exception as e:
            print(f"Error in LLM planning: {e}")
            return None

    def _construct_user_message(self, request: PlanningRequest) -> str:
        """Construct the user message for the LLM"""
        message = f"""
        Goal: {request.goal}

        Current Robot State:
        - Position: {request.current_state.position}
        - Battery: {request.current_state.battery_level}%
        - Available Actions: {request.current_state.available_actions}
        - Current Task: {request.current_state.current_task}
        - Detected Objects: {request.current_state.detected_objects}

        Context:
        {json.dumps(request.context, indent=2)}

        Constraints:
        {chr(10).join(request.constraints)}

        Please provide a detailed plan to achieve this goal, considering the robot's current state and constraints.
        """

        return message

    def _parse_response(self, content: str) -> Optional[PlanningResponse]:
        """Parse the LLM response into a structured format"""
        try:
            # Extract JSON from the response (in case there's additional text)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)

                return PlanningResponse(
                    plan=data.get('plan', []),
                    confidence=data.get('confidence', 0.5),
                    reasoning=data.get('reasoning', ''),
                    estimated_time=data.get('estimated_time', 0.0)
                )
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response content: {content}")

        return None

# Example usage
async def example_usage():
    # Initialize the planner
    planner = LLMPlanner(api_key="your-api-key")

    # Create a sample request
    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=85.0,
        available_actions=["move_to", "pick_up", "put_down", "wave", "speak"],
        current_task=None,
        detected_objects=[{"name": "bottle", "position": {"x": 1.0, "y": 0.5}}],
        human_interactions=[]
    )

    request = PlanningRequest(
        goal="Pick up the bottle and bring it to the kitchen",
        current_state=state,
        context={"locations": {"kitchen": {"x": 2.0, "y": 1.0}}},
        constraints=["avoid obstacles", "preserve battery"]
    )

    # Generate plan
    response = await planner.plan_task(request)
    if response:
        print(f"Generated plan: {response.plan}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Confidence: {response.confidence}")

if __name__ == "__main__":
    asyncio.run(example_usage())
```

### Caching and Optimization Pattern

```python
# llm_cache.py
import hashlib
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class CacheEntry:
    """Entry in the LLM response cache"""
    request_hash: str
    response: Dict[str, Any]
    timestamp: float
    ttl: float  # Time to live in seconds

class LLMCache:
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def _generate_hash(self, request: Dict[str, Any]) -> str:
        """Generate a hash for the request to use as cache key"""
        request_str = json.dumps(request, sort_keys=True, default=str)
        return hashlib.sha256(request_str.encode()).hexdigest()

    def get(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for request"""
        request_hash = self._generate_hash(request)

        if request_hash in self.cache:
            entry = self.cache[request_hash]

            # Check if entry is still valid
            if time.time() - entry.timestamp < entry.ttl:
                print(f"Cache hit for request: {request_hash[:8]}...")
                return entry.response
            else:
                # Remove expired entry
                del self.cache[request_hash]

        return None

    def put(self, request: Dict[str, Any], response: Dict[str, Any], ttl: Optional[float] = None):
        """Put response in cache"""
        request_hash = self._generate_hash(request)

        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]

        entry = CacheEntry(
            request_hash=request_hash,
            response=response,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl
        )

        self.cache[request_hash] = entry
        print(f"Cache miss, stored response for request: {request_hash[:8]}...")

    def clear_expired(self):
        """Remove all expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp >= entry.ttl
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

# Example usage with caching
class CachedLLMPlanner(LLMPlanner):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", cache_ttl: float = 3600.0):
        super().__init__(api_key, model)
        self.cache = LLMCache(default_ttl=cache_ttl)

    async def plan_task(self, request: PlanningRequest) -> Optional[PlanningResponse]:
        # Create cache key from request
        cache_key = {
            'goal': request.goal,
            'available_actions': request.current_state.available_actions,
            'constraints': request.constraints
        }

        # Try to get from cache first
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return PlanningResponse(**cached_response)

        # If not in cache, call the original method
        response = await super().plan_task(request)

        if response:
            # Store in cache
            cache_value = asdict(response)
            self.cache.put(cache_key, cache_value)

        return response
```

## Task Planning with LLMs

### Hierarchical Task Planning

```python
# hierarchical_planning.py
from typing import List, Dict, Any, Optional
import asyncio

class HierarchicalPlanner:
    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner
        self.task_decomposition_rules = {
            "complex_navigation": ["find_path", "avoid_obstacles", "reach_destination"],
            "object_manipulation": ["locate_object", "approach_object", "grasp_object", "transport_object", "place_object"],
            "social_interaction": ["detect_human", "approach_human", "greet_human", "engage_conversation", "farewell"]
        }

    async def decompose_task(self, high_level_goal: str) -> List[Dict[str, Any]]:
        """Decompose a high-level goal into subtasks"""
        # First, try to match against known patterns
        subtasks = self._match_decomposition_pattern(high_level_goal)

        if not subtasks:
            # If no pattern matches, ask LLM to decompose
            subtasks = await self._llm_decompose(high_level_goal)

        return subtasks

    def _match_decomposition_pattern(self, goal: str) -> List[Dict[str, Any]]:
        """Match goal against known decomposition patterns"""
        goal_lower = goal.lower()

        for pattern, subtasks in self.task_decomposition_rules.items():
            if pattern in goal_lower:
                return [{"action": subtask, "parameters": {}} for subtask in subtasks]

        return []

    async def _llm_decompose(self, goal: str) -> List[Dict[str, Any]]:
        """Ask LLM to decompose the goal into subtasks"""
        prompt = f"""
        Decompose the following goal into logical subtasks that a humanoid robot can execute:
        Goal: {goal}

        Return a list of subtasks in JSON format: [{{"action": string, "parameters": {{}}}}, ...]
        Each subtask should be a specific, executable action.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.llm_planner.model,
                messages=[
                    {"role": "system", "content": "You are a task decomposition expert for humanoid robots. Break down goals into executable subtasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            content = response.choices[0].message.content
            # Extract JSON from response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1

            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Error in LLM task decomposition: {e}")

        # Return a default decomposition if LLM fails
        return [{"action": "unknown", "parameters": {"goal": goal}}]

    async def generate_detailed_plan(self, high_level_goal: str, robot_state: RobotState) -> List[Dict[str, Any]]:
        """Generate a detailed execution plan for the high-level goal"""
        # Decompose the goal
        subtasks = await self.decompose_task(high_level_goal)

        detailed_plan = []

        for i, subtask in enumerate(subtasks):
            # For each subtask, generate detailed execution steps
            request = PlanningRequest(
                goal=f"Execute subtask: {subtask['action']}",
                current_state=robot_state,
                context={"subtask_index": i, "total_subtasks": len(subtasks)},
                constraints=[]
            )

            response = await self.llm_planner.plan_task(request)
            if response and response.plan:
                detailed_plan.extend(response.plan)

        return detailed_plan

# Example usage
async def hierarchical_example():
    planner = LLMPlanner(api_key="your-api-key")
    hierarchical_planner = HierarchicalPlanner(planner)

    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=85.0,
        available_actions=["move_to", "pick_up", "put_down", "wave", "speak"],
        current_task=None,
        detected_objects=[],
        human_interactions=[]
    )

    goal = "Navigate to kitchen, pick up a bottle, and bring it to the living room"
    detailed_plan = await hierarchical_planner.generate_detailed_plan(goal, state)

    print(f"Generated detailed plan with {len(detailed_plan)} steps:")
    for i, step in enumerate(detailed_plan):
        print(f"  {i+1}. {step['action']} with params {step.get('parameters', {})}")
```

### Context-Aware Planning

```python
# context_aware_planning.py
import datetime
from typing import Dict, Any, List
import asyncio

class ContextAwarePlanner:
    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner
        self.context_history = []
        self.max_context_length = 50  # Maximum number of context entries

    def update_context(self, event: Dict[str, Any]):
        """Update the context with a new event"""
        event['timestamp'] = datetime.datetime.now().isoformat()
        self.context_history.append(event)

        # Keep context history within limits
        if len(self.context_history) > self.max_context_length:
            self.context_history = self.context_history[-self.max_context_length:]

    def get_relevant_context(self, goal: str, window_size: int = 10) -> List[Dict[str, Any]]:
        """Get relevant context for the current goal"""
        # For now, return the most recent context entries
        # In practice, you might want to implement semantic search
        relevant = self.context_history[-window_size:]

        # Add current time and day context
        current_time = datetime.datetime.now()
        relevant.append({
            'type': 'time_context',
            'hour': current_time.hour,
            'day_of_week': current_time.strftime('%A'),
            'is_weekend': current_time.weekday() >= 5
        })

        return relevant

    async def plan_with_context(self, goal: str, robot_state: RobotState) -> Optional[PlanningResponse]:
        """Plan with consideration of historical context"""
        # Get relevant context
        context = self.get_relevant_context(goal)

        # Create planning request with context
        request = PlanningRequest(
            goal=goal,
            current_state=robot_state,
            context={
                'historical_context': context,
                'current_time': datetime.datetime.now().isoformat()
            },
            constraints=[]
        )

        return await self.llm_planner.plan_task(request)

    def learn_from_execution(self, goal: str, plan: List[Dict[str, Any]],
                           success: bool, feedback: str = ""):
        """Learn from plan execution results"""
        learning_event = {
            'type': 'execution_result',
            'goal': goal,
            'plan': plan,
            'success': success,
            'feedback': feedback,
            'timestamp': datetime.datetime.now().isoformat()
        }

        self.update_context(learning_event)

# Example usage
async def context_aware_example():
    planner = LLMPlanner(api_key="your-api-key")
    context_planner = ContextAwarePlanner(planner)

    # Simulate some context
    context_planner.update_context({
        'type': 'human_interaction',
        'person': 'John',
        'action': 'requested_help',
        'location': 'kitchen'
    })

    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=85.0,
        available_actions=["move_to", "pick_up", "put_down", "wave", "speak"],
        current_task=None,
        detected_objects=[],
        human_interactions=[]
    )

    goal = "Help John in the kitchen"
    response = await context_planner.plan_with_context(goal, state)

    if response:
        print(f"Context-aware plan: {response.plan}")
        print(f"Reasoning: {response.reasoning}")
```

## Context and Memory Management

### Memory System for Long-term Context

```python
# memory_system.py
import sqlite3
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class RobotMemory:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the memory database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # This allows accessing columns by name

        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                access_count INTEGER DEFAULT 0
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id_1 INTEGER,
                memory_id_2 INTEGER,
                relationship_type TEXT,
                strength REAL DEFAULT 1.0,
                FOREIGN KEY (memory_id_1) REFERENCES memories (id),
                FOREIGN KEY (memory_id_2) REFERENCES memories (id)
            )
        ''')

        self.conn.commit()

    def store_memory(self, memory_type: str, content: Dict[str, Any],
                     tags: List[str] = None, importance: float = 0.5) -> int:
        """Store a memory in the database"""
        content_json = json.dumps(content)
        tags_json = json.dumps(tags) if tags else "[]"

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO memories (memory_type, content, tags, importance)
            VALUES (?, ?, ?, ?)
        ''', (memory_type, content_json, tags_json, importance))

        memory_id = cursor.lastrowid
        self.conn.commit()

        return memory_id

    def retrieve_memories(self, memory_type: str = None, tags: List[str] = None,
                         time_window_hours: int = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memories based on criteria"""
        query = "SELECT * FROM memories WHERE 1=1"
        params = []

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        if tags:
            # Search for memories that contain any of the specified tags
            query += " AND ("
            tag_conditions = []
            for i, tag in enumerate(tags):
                tag_conditions.append(f"tags LIKE ?")
                params.append(f'%"{tag}"%')
            query += " OR ".join(tag_conditions) + ")"

        if time_window_hours:
            query += " AND timestamp > datetime('now', ?)"
            params.append(f"-{time_window_hours} hours")

        query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            memory = dict(row)
            memory['content'] = json.loads(memory['content'])
            memory['tags'] = json.loads(memory['tags'])
            results.append(memory)

        return results

    def update_memory_access(self, memory_id: int):
        """Update the access count for a memory"""
        self.conn.execute('''
            UPDATE memories
            SET access_count = access_count + 1
            WHERE id = ?
        ''', (memory_id,))
        self.conn.commit()

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories relevant to a query (simplified - in practice use semantic search)"""
        # This is a simplified version; in practice, you'd use embeddings and similarity search
        # For now, we'll search in content text

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM memories
            WHERE content LIKE ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        results = []
        for row in cursor.fetchall():
            memory = dict(row)
            memory['content'] = json.loads(memory['content'])
            memory['tags'] = json.loads(memory['tags'])
            results.append(memory)

        return results

    def create_relationship(self, memory_id_1: int, memory_id_2: int,
                          relationship_type: str, strength: float = 1.0):
        """Create a relationship between two memories"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO relationships (memory_id_1, memory_id_2, relationship_type, strength)
            VALUES (?, ?, ?, ?)
        ''', (memory_id_1, memory_id_2, relationship_type, strength))
        self.conn.commit()

    def get_memory_graph(self, memory_id: int, depth: int = 2) -> List[Dict[str, Any]]:
        """Get related memories in a graph structure"""
        # This would return a graph of related memories
        # Simplified implementation - in practice, use recursive queries
        cursor = self.conn.cursor()

        # Get direct relationships
        cursor.execute('''
            SELECT r.*, m1.content as memory1_content, m2.content as memory2_content
            FROM relationships r
            JOIN memories m1 ON r.memory_id_1 = m1.id
            JOIN memories m2 ON r.memory_id_2 = m2.id
            WHERE r.memory_id_1 = ? OR r.memory_id_2 = ?
        ''', (memory_id, memory_id))

        relationships = []
        for row in cursor.fetchall():
            rel = dict(row)
            rel['memory1_content'] = json.loads(rel['memory1_content'])
            rel['memory2_content'] = json.loads(rel['memory2_content'])
            relationships.append(rel)

        return relationships

class ContextManager:
    def __init__(self, memory_system: RobotMemory):
        self.memory_system = memory_system
        self.current_context = {}
        self.context_stack = []

    def push_context(self, context: Dict[str, Any]):
        """Push a new context onto the stack"""
        self.context_stack.append(self.current_context.copy())
        self.current_context.update(context)

    def pop_context(self):
        """Pop the current context and return to previous"""
        if self.context_stack:
            self.current_context = self.context_stack.pop()

    def add_to_context(self, key: str, value: Any):
        """Add a value to the current context"""
        self.current_context[key] = value

    def get_context(self) -> Dict[str, Any]:
        """Get the current context"""
        return self.current_context.copy()

    def store_context_as_memory(self, context_type: str, tags: List[str] = None):
        """Store the current context as a memory"""
        memory_id = self.memory_system.store_memory(
            memory_type=context_type,
            content=self.current_context,
            tags=tags
        )
        return memory_id

    def load_context_from_memory(self, memory_id: int):
        """Load context from a stored memory"""
        cursor = self.memory_system.conn.cursor()
        cursor.execute('SELECT content FROM memories WHERE id = ?', (memory_id,))
        row = cursor.fetchone()

        if row:
            content = json.loads(row['content'])
            self.current_context.update(content)
            self.memory_system.update_memory_access(memory_id)

# Example usage
def memory_example():
    memory = RobotMemory()
    context_manager = ContextManager(memory)

    # Store some memories
    memory.store_memory(
        memory_type="human_interaction",
        content={
            "person": "John",
            "action": "requested_help",
            "location": "kitchen",
            "time": "2023-10-15T10:30:00"
        },
        tags=["help", "kitchen", "morning"],
        importance=0.8
    )

    memory.store_memory(
        memory_type="object_location",
        content={
            "object": "bottle",
            "location": "kitchen_counter",
            "confidence": 0.9
        },
        tags=["object", "location", "kitchen"],
        importance=0.6
    )

    # Retrieve relevant memories
    interactions = memory.retrieve_memories(memory_type="human_interaction", limit=5)
    print(f"Retrieved {len(interactions)} human interactions")

    # Use context manager
    context_manager.add_to_context("current_user", "John")
    context_manager.add_to_context("current_task", "help_request")

    print(f"Current context: {context_manager.get_context()}")
```

## Perception-Action Integration

### Perception-to-Action Pipeline

```python
# perception_action.py
from typing import Dict, List, Any, Optional
import asyncio
import time

class PerceptionActionBridge:
    def __init__(self, llm_planner: LLMPlanner, memory_system: RobotMemory):
        self.llm_planner = llm_planner
        self.memory_system = memory_system
        self.perception_handlers = {}
        self.action_executors = {}

    def register_perception_handler(self, sensor_type: str, handler_func):
        """Register a handler for a specific sensor type"""
        self.perception_handlers[sensor_type] = handler_func

    def register_action_executor(self, action_type: str, executor_func):
        """Register an executor for a specific action type"""
        self.action_executors[action_type] = executor_func

    async def process_perception(self, sensor_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process perception data and potentially trigger actions"""
        if sensor_type not in self.perception_handlers:
            print(f"No handler for sensor type: {sensor_type}")
            return None

        # Process the perception data
        processed_data = await self.perception_handlers[sensor_type](data)

        # Store in memory
        memory_id = self.memory_system.store_memory(
            memory_type=f"perception_{sensor_type}",
            content=processed_data,
            tags=["perception", sensor_type],
            importance=0.7
        )

        return processed_data

    async def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute an action using the appropriate executor"""
        action_type = action.get('action', 'unknown')

        if action_type not in self.action_executors:
            print(f"No executor for action type: {action_type}")
            return False

        try:
            success = await self.action_executors[action_type](action.get('parameters', {}))

            # Log the action execution
            self.memory_system.store_memory(
                memory_type="action_execution",
                content={
                    "action": action_type,
                    "parameters": action.get('parameters', {}),
                    "success": success,
                    "timestamp": time.time()
                },
                tags=["action", action_type, "execution"],
                importance=0.5
            )

            return success
        except Exception as e:
            print(f"Error executing action {action_type}: {e}")
            return False

    async def run_perception_action_cycle(self, robot_state: RobotState):
        """Run a complete perception-action cycle"""
        # Get relevant memories to inform perception
        recent_memories = self.memory_system.retrieve_memories(
            time_window_hours=1,
            limit=20
        )

        # Process each sensor input
        sensor_inputs = await self.get_sensor_inputs()
        for sensor_type, data in sensor_inputs.items():
            processed = await self.process_perception(sensor_type, data)

            if processed:
                # Check if this perception should trigger planning
                should_plan = await self.should_trigger_planning(processed, robot_state)

                if should_plan:
                    # Generate a plan based on the perception
                    plan = await self.generate_perception_based_plan(processed, robot_state)

                    if plan:
                        # Execute the plan
                        for action in plan:
                            success = await self.execute_action(action)
                            if not success:
                                print(f"Action execution failed: {action}")
                                break

    async def get_sensor_inputs(self) -> Dict[str, Any]:
        """Get inputs from all sensors (placeholder implementation)"""
        # In practice, this would interface with actual sensors
        return {
            "camera": {"objects": ["person", "bottle"], "locations": {"person": [1.0, 2.0], "bottle": [0.5, 1.5]}},
            "lidar": {"obstacles": [{"distance": 1.2, "angle": 45}]},
            "microphone": {"speech": "Could you help me?", "confidence": 0.85}
        }

    async def should_trigger_planning(self, perception_data: Dict[str, Any],
                                    robot_state: RobotState) -> bool:
        """Determine if planning should be triggered based on perception"""
        # Example: Plan if a human is detected asking for help
        if perception_data.get('sensor_type') == 'microphone':
            speech = perception_data.get('speech', '')
            if any(phrase in speech.lower() for phrase in ['help', 'assist', 'can you']):
                return True

        # Example: Plan if a new object is detected that matches current goal
        if perception_data.get('sensor_type') == 'camera':
            detected_objects = perception_data.get('objects', [])
            if 'bottle' in detected_objects and robot_state.current_task == 'find_bottle':
                return True

        return False

    async def generate_perception_based_plan(self, perception_data: Dict[str, Any],
                                           robot_state: RobotState) -> Optional[List[Dict[str, Any]]]:
        """Generate a plan based on perception data"""
        # Create a goal based on perception
        if perception_data.get('sensor_type') == 'microphone':
            speech = perception_data.get('speech', '')
            goal = f"Respond to human request: {speech}"
        elif perception_data.get('sensor_type') == 'camera':
            detected_objects = perception_data.get('objects', [])
            if detected_objects:
                goal = f"Interact with detected objects: {detected_objects}"

        # Use LLM to plan the response
        request = PlanningRequest(
            goal=goal,
            current_state=robot_state,
            context={"perception_data": perception_data},
            constraints=[]
        )

        response = await self.llm_planner.plan_task(request)
        return response.plan if response else None

# Example perception handlers
async def camera_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle camera perception data"""
    objects = data.get('objects', [])
    locations = data.get('locations', {})

    return {
        "sensor_type": "camera",
        "objects_detected": len(objects),
        "object_list": objects,
        "locations": locations,
        "timestamp": time.time()
    }

async def lidar_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle LiDAR perception data"""
    obstacles = data.get('obstacles', [])

    return {
        "sensor_type": "lidar",
        "obstacles_detected": len(obstacles),
        "obstacle_list": obstacles,
        "timestamp": time.time()
    }

async def microphone_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle microphone perception data"""
    speech = data.get('speech', '')
    confidence = data.get('confidence', 0.0)

    return {
        "sensor_type": "microphone",
        "transcribed_speech": speech,
        "confidence": confidence,
        "timestamp": time.time()
    }

# Example action executors
async def move_to_executor(params: Dict[str, Any]) -> bool:
    """Execute move to location action"""
    location = params.get('location', {})
    print(f"Moving to location: {location}")
    # In practice, this would interface with navigation system
    time.sleep(1)  # Simulate execution time
    return True

async def pick_up_executor(params: Dict[str, Any]) -> bool:
    """Execute pick up object action"""
    object_name = params.get('object', 'unknown')
    print(f"Attempting to pick up: {object_name}")
    # In practice, this would interface with manipulation system
    time.sleep(2)  # Simulate execution time
    return True

async def speak_executor(params: Dict[str, Any]) -> bool:
    """Execute speak action"""
    text = params.get('text', '')
    print(f"Speaking: {text}")
    # In practice, this would interface with TTS system
    time.sleep(1)  # Simulate execution time
    return True

# Example usage
async def perception_action_example():
    planner = LLMPlanner(api_key="your-api-key")
    memory = RobotMemory()
    bridge = PerceptionActionBridge(planner, memory)

    # Register handlers and executors
    bridge.register_perception_handler("camera", camera_handler)
    bridge.register_perception_handler("lidar", lidar_handler)
    bridge.register_perception_handler("microphone", microphone_handler)

    bridge.register_action_executor("move_to", move_to_executor)
    bridge.register_action_executor("pick_up", pick_up_executor)
    bridge.register_action_executor("speak", speak_executor)

    # Create initial robot state
    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=85.0,
        available_actions=["move_to", "pick_up", "speak"],
        current_task="idle",
        detected_objects=[],
        human_interactions=[]
    )

    # Run a perception-action cycle
    await bridge.run_perception_action_cycle(state)
```

## Real-time Considerations

### Optimized Real-time Processing

```python
# real_time_optimization.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable
import threading

class RealTimeLLMPlanner:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo",
                 max_concurrent_requests: int = 3):
        self.api_key = api_key
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests

        # Thread pool for non-blocking operations
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)

        # Request queue for prioritization
        self.request_queue = asyncio.Queue()
        self.response_callbacks = {}

        # Performance metrics
        self.metrics = {
            'request_count': 0,
            'avg_response_time': 0.0,
            'failed_requests': 0,
            'cached_responses': 0
        }

        # Caching system
        self.cache = LLMCache()

        # Start processing loop
        self.processing_task = None
        self.running = False

    async def start_processing(self):
        """Start the real-time processing loop"""
        self.running = True
        self.processing_task = asyncio.create_task(self._processing_loop())

    async def stop_processing(self):
        """Stop the real-time processing"""
        self.running = False
        if self.processing_task:
            await self.processing_task

    async def _processing_loop(self):
        """Main processing loop for handling requests"""
        while self.running:
            try:
                # Get request from queue
                request_id, request, callback = await asyncio.wait_for(
                    self.request_queue.get(), timeout=0.1
                )

                # Process the request
                response = await self._process_request(request)

                # Call the callback with the response
                if callback:
                    await callback(response)

                self.request_queue.task_done()

            except asyncio.TimeoutError:
                continue  # No requests, continue loop
            except Exception as e:
                print(f"Error in processing loop: {e}")

    async def _process_request(self, request: PlanningRequest) -> Optional[PlanningResponse]:
        """Process a single planning request with optimization"""
        start_time = time.time()

        try:
            # Try cache first
            cache_key = self._create_cache_key(request)
            cached_response = self.cache.get(cache_key)

            if cached_response:
                self.metrics['cached_responses'] += 1
                return PlanningResponse(**cached_response)

            # For real-time applications, use a faster but less detailed approach
            # This could involve using a smaller model or simplified prompts
            response = await self._optimized_plan_task(request)

            if response:
                # Cache the response
                cache_value = {
                    'plan': response.plan,
                    'confidence': response.confidence,
                    'reasoning': response.reasoning,
                    'estimated_time': response.estimated_time
                }
                self.cache.put(cache_key, cache_value)

            # Update metrics
            response_time = time.time() - start_time
            self.metrics['request_count'] += 1
            total_time = self.metrics['avg_response_time'] * (self.metrics['request_count'] - 1) + response_time
            self.metrics['avg_response_time'] = total_time / self.metrics['request_count']

            return response

        except Exception as e:
            self.metrics['failed_requests'] += 1
            print(f"Error processing request: {e}")
            return None

    def _create_cache_key(self, request: PlanningRequest) -> Dict[str, Any]:
        """Create a cache key for the request"""
        return {
            'goal': request.goal,
            'available_actions': tuple(sorted(request.current_state.available_actions)),
            'constraints': tuple(sorted(request.constraints))
        }

    async def _optimized_plan_task(self, request: PlanningRequest) -> Optional[PlanningResponse]:
        """Optimized planning for real-time applications"""
        # Use a more focused prompt for faster response
        focused_prompt = f"""
        Goal: {request.goal}
        Available actions: {request.current_state.available_actions}
        Constraints: {', '.join(request.constraints)}

        Provide a concise plan as JSON: {{"plan": [{{"action": string, "parameters": {{}}}}], "reasoning": string, "confidence": float}}
        """

        try:
            # Use a timeout for the LLM call to ensure real-time constraints
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    lambda: openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a fast robotic planning assistant. Provide concise plans."},
                            {"role": "user", "content": focused_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=300
                    )
                ),
                timeout=5.0  # 5 second timeout for real-time applications
            )

            content = response.choices[0].message.content
            return self._parse_response(content)

        except asyncio.TimeoutError:
            print("LLM request timed out")
            return None
        except Exception as e:
            print(f"Error in optimized planning: {e}")
            return None

    async def plan_task_async(self, request: PlanningRequest,
                            callback: Optional[Callable] = None) -> str:
        """Add a planning request to the queue"""
        request_id = f"req_{int(time.time() * 1000000)}"
        await self.request_queue.put((request_id, request, callback))
        return request_id

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

class PrioritizedRequestQueue:
    """A queue that prioritizes requests based on urgency"""

    def __init__(self):
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()

    async def put(self, item: Any, priority: str = "normal"):
        """Put an item in the queue with specified priority"""
        queue_map = {
            "high": self.high_priority_queue,
            "normal": self.normal_priority_queue,
            "low": self.low_priority_queue
        }

        await queue_map.get(priority, self.normal_priority_queue).put(item)

    async def get(self):
        """Get an item from the queue, prioritizing higher priority queues"""
        # Try high priority first
        try:
            return await asyncio.wait_for(self.high_priority_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            pass

        # Try normal priority
        try:
            return await asyncio.wait_for(self.normal_priority_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            pass

        # Finally, low priority
        return await self.low_priority_queue.get()

# Example usage
async def real_time_example():
    planner = RealTimeLLMPlanner(api_key="your-api-key")
    await planner.start_processing()

    try:
        # Create a sample request
        state = RobotState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=85.0,
            available_actions=["move_to", "pick_up", "speak"],
            current_task=None,
            detected_objects=[],
            human_interactions=[]
        )

        request = PlanningRequest(
            goal="Move to the kitchen",
            current_state=state,
            context={},
            constraints=[]
        )

        # Plan with callback
        async def callback(response):
            if response:
                print(f"Received plan: {response.plan}")
            else:
                print("Failed to get plan")

        request_id = await planner.plan_task_async(request, callback)
        print(f"Submitted request {request_id}")

        # Wait a bit for processing
        await asyncio.sleep(2)

        # Check metrics
        metrics = planner.get_performance_metrics()
        print(f"Performance metrics: {metrics}")

    finally:
        await planner.stop_processing()
```

## Uncertainty Handling

### Uncertainty-Aware Planning

```python
# uncertainty_handling.py
import random
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class UncertaintyAwarePlanner:
    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner
        self.uncertainty_models = {}
        self.belief_state = {}

    def update_belief(self, observation: Dict[str, Any], confidence: float):
        """Update the robot's belief state based on observation"""
        for key, value in observation.items():
            if key not in self.belief_state:
                self.belief_state[key] = {'value': value, 'confidence': confidence}
            else:
                # Update using weighted average based on confidence
                old_confidence = self.belief_state[key]['confidence']
                old_value = self.belief_state[key]['value']

                new_confidence = min(1.0, old_confidence + confidence * 0.1)  # Confidence increases slowly
                weighted_value = (old_value * old_confidence + value * confidence) / (old_confidence + confidence)

                self.belief_state[key] = {'value': weighted_value, 'confidence': new_confidence}

    def estimate_uncertainty(self, goal: str, current_state: RobotState) -> Dict[str, float]:
        """Estimate uncertainty for a given goal and state"""
        uncertainty_estimates = {
            'navigation_uncertainty': 0.1,  # Base uncertainty
            'object_detection_uncertainty': 0.2,
            'action_success_uncertainty': 0.15,
            'temporal_uncertainty': 0.05
        }

        # Increase uncertainty based on environmental factors
        if current_state.battery_level < 20:
            uncertainty_estimates['action_success_uncertainty'] += 0.2

        # Increase uncertainty if many objects detected (cluttered environment)
        if len(current_state.detected_objects) > 5:
            uncertainty_estimates['navigation_uncertainty'] += 0.15

        return uncertainty_estimates

    async def plan_with_uncertainty(self, goal: str, current_state: RobotState) -> Optional[PlanningResponse]:
        """Plan considering uncertainty in the environment"""
        # Estimate current uncertainties
        uncertainties = self.estimate_uncertainty(goal, current_state)

        # Create a robust planning request that accounts for uncertainties
        request = PlanningRequest(
            goal=goal,
            current_state=current_state,
            context={
                'uncertainties': uncertainties,
                'belief_state': self.belief_state
            },
            constraints=[
                f"Account for navigation uncertainty: {uncertainties['navigation_uncertainty']}",
                f"Account for object detection uncertainty: {uncertainties['object_detection_uncertainty']}",
                f"Account for action success uncertainty: {uncertainties['action_success_uncertainty']}"
            ]
        )

        # Get initial plan
        response = await self.llm_planner.plan_task(request)

        if response:
            # Add uncertainty handling to the plan
            enhanced_plan = self._add_uncertainty_handling(response.plan, uncertainties)
            response.plan = enhanced_plan

        return response

    def _add_uncertainty_handling(self, plan: List[Dict[str, Any]],
                                uncertainties: Dict[str, float]) -> List[Dict[str, Any]]:
        """Add uncertainty handling steps to the plan"""
        enhanced_plan = []

        for step in plan:
            enhanced_plan.append(step)

            # Add verification steps based on uncertainty levels
            if uncertainties['object_detection_uncertainty'] > 0.15:
                # Add verification after object-related actions
                if any(obj in str(step) for obj in ['pick_up', 'put_down', 'locate']):
                    enhanced_plan.append({
                        'action': 'verify_object_detection',
                        'parameters': {'timeout': 2.0}
                    })

            # Add navigation verification for high navigation uncertainty
            if uncertainties['navigation_uncertainty'] > 0.15:
                if 'move_to' in str(step):
                    enhanced_plan.append({
                        'action': 'verify_position',
                        'parameters': {'tolerance': 0.2}
                    })

        return enhanced_plan

class ProbabilisticActionExecutor:
    """Executes actions with probabilistic success rates"""

    def __init__(self):
        self.action_success_rates = {
            'move_to': 0.95,
            'pick_up': 0.85,
            'put_down': 0.98,
            'speak': 0.99,
            'wave': 0.97
        }

    async def execute_with_probability(self, action: Dict[str, Any]) -> Tuple[bool, float]:
        """Execute action and return success status with confidence"""
        action_type = action.get('action', 'unknown')

        # Get base success rate
        base_success_rate = self.action_success_rates.get(action_type, 0.9)

        # Modify based on parameters and context
        parameters = action.get('parameters', {})

        # For navigation, success rate might depend on distance
        if action_type == 'move_to':
            distance = parameters.get('distance', 1.0)
            if distance > 5.0:  # Long distance navigation
                base_success_rate *= 0.8

        # For manipulation, success rate might depend on object properties
        if action_type in ['pick_up', 'place']:
            obj_size = parameters.get('object_size', 'medium')
            if obj_size == 'small':
                base_success_rate *= 0.7
            elif obj_size == 'large':
                base_success_rate *= 0.9

        # Determine success based on probability
        success = random.random() < base_success_rate
        confidence = base_success_rate if success else (1 - base_success_rate)

        return success, confidence

class AdaptivePlanner:
    """Adapts plans based on execution feedback"""

    def __init__(self, uncertainty_planner: UncertaintyAwarePlanner):
        self.uncertainty_planner = uncertainty_planner
        self.execution_history = []
        self.adaptation_rules = []

    def record_execution_result(self, action: Dict[str, Any], success: bool,
                              confidence: float, execution_time: float):
        """Record the result of an action execution"""
        result = {
            'action': action,
            'success': success,
            'confidence': confidence,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        self.execution_history.append(result)

    def adapt_plan(self, original_plan: List[Dict[str, Any]],
                  execution_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adapt the plan based on execution feedback"""
        adapted_plan = original_plan.copy()

        # Analyze feedback to identify problematic actions
        failed_actions = [f for f in execution_feedback if not f['success']]

        for failed_action in failed_actions:
            action_index = self._find_action_in_plan(adapted_plan, failed_action['action'])
            if action_index != -1:
                # Replace failed action with alternative or add recovery
                adapted_plan = self._adapt_action(adapted_plan, action_index, failed_action)

        return adapted_plan

    def _find_action_in_plan(self, plan: List[Dict[str, Any]], target_action: Dict[str, Any]) -> int:
        """Find the index of an action in the plan"""
        for i, action in enumerate(plan):
            if action['action'] == target_action['action']:
                # Check if parameters match closely enough
                target_params = target_action.get('parameters', {})
                action_params = action.get('parameters', {})

                if self._parameters_match(target_params, action_params):
                    return i
        return -1

    def _parameters_match(self, params1: Dict[str, Any], params2: Dict[str, Any],
                         threshold: float = 0.8) -> bool:
        """Check if action parameters match above a threshold"""
        if not params1 and not params2:
            return True

        # Simple matching - in practice, you'd want more sophisticated comparison
        matching_keys = set(params1.keys()) & set(params2.keys())
        total_keys = set(params1.keys()) | set(params2.keys())

        if not total_keys:
            return True

        match_ratio = len(matching_keys) / len(total_keys)
        return match_ratio >= threshold

    def _adapt_action(self, plan: List[Dict[str, Any]], action_index: int,
                     failure_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt a specific action in the plan"""
        adapted_plan = plan.copy()
        failed_action = adapted_plan[action_index]

        # Different adaptation strategies based on action type
        action_type = failed_action['action']

        if action_type == 'move_to':
            # Try alternative navigation method or path
            adapted_plan[action_index] = {
                'action': 'move_to_alternative',
                'parameters': {**failed_action['parameters'], 'retry_count': 1}
            }
        elif action_type == 'pick_up':
            # Try different grasp or approach
            adapted_plan[action_index] = {
                'action': 'pick_up_alternative',
                'parameters': {**failed_action['parameters'], 'retry_count': 1}
            }
        elif action_type == 'navigate':
            # Add more careful navigation with verification steps
            verification_step = {
                'action': 'verify_navigation',
                'parameters': {'timeout': 5.0}
            }
            adapted_plan.insert(action_index + 1, verification_step)

        return adapted_plan

# Example usage
async def uncertainty_example():
    llm_planner = LLMPlanner(api_key="your-api-key")
    uncertainty_planner = UncertaintyAwarePlanner(llm_planner)
    adaptive_planner = AdaptivePlanner(uncertainty_planner)

    # Update belief with some observations
    uncertainty_planner.update_belief(
        {'battery_level': 65, 'object_ahead': 'bottle'},
        confidence=0.8
    )

    # Create robot state
    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=65.0,
        available_actions=["move_to", "pick_up", "speak"],
        current_task=None,
        detected_objects=[{"name": "bottle", "confidence": 0.8}],
        human_interactions=[]
    )

    # Plan with uncertainty
    response = await uncertainty_planner.plan_with_uncertainty("Pick up the bottle", state)

    if response:
        print(f"Uncertainty-aware plan: {response.plan}")
        print(f"Reasoning: {response.reasoning}")

        # Simulate execution and adaptation
        executor = ProbabilisticActionExecutor()

        for i, action in enumerate(response.plan):
            success, confidence = await executor.execute_with_probability(action)
            print(f"Action {i+1}: {action['action']} - Success: {success}, Confidence: {confidence:.2f}")

            # Record result for adaptation
            adaptive_planner.record_execution_result(
                action, success, confidence, random.uniform(1.0, 3.0)
            )
```

## Safety and Ethics

### Safety-Aware Planning

```python
# safety_ethics.py
from typing import Dict, List, Any, Optional
import asyncio

class SafetyChecker:
    def __init__(self):
        self.safety_rules = [
            # Physical safety rules
            lambda action, state: self._check_collision_risk(action, state),
            lambda action, state: self._check_battery_safety(action, state),
            lambda action, state: self._check_manipulation_safety(action, state),

            # Social safety rules
            lambda action, state: self._check_personal_space(action, state),
            lambda action, state: self._check_consent(action, state),
        ]

    def _check_collision_risk(self, action: Dict[str, Any], state: RobotState) -> Dict[str, Any]:
        """Check for potential collision risks"""
        if action['action'] == 'move_to':
            target_pos = action.get('parameters', {}).get('position', {})
            current_pos = state.position

            # Check if movement is toward a detected object
            for obj in state.detected_objects:
                obj_pos = obj.get('position', {})
                if self._is_collision_path(current_pos, target_pos, obj_pos):
                    return {
                        'safe': False,
                        'reason': 'Potential collision with detected object',
                        'severity': 'high'
                    }

        return {'safe': True, 'reason': 'No collision risk detected', 'severity': 'low'}

    def _is_collision_path(self, start: Dict[str, float], end: Dict[str, float],
                          obstacle: Dict[str, float], threshold: float = 0.3) -> bool:
        """Check if path from start to end passes near obstacle"""
        # Simplified collision detection
        # Calculate distance from line segment to point
        start_vec = np.array([start['x'], start['y']])
        end_vec = np.array([end['x'], end['y']])
        obstacle_vec = np.array([obstacle['x'], obstacle['y']])

        # Calculate distance from point to line segment
        line_vec = end_vec - start_vec
        point_vec = obstacle_vec - start_vec

        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point_vec) < threshold

        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = start_vec + t * line_vec
        distance = np.linalg.norm(obstacle_vec - projection)

        return distance < threshold

    def _check_battery_safety(self, action: Dict[str, Any], state: RobotState) -> Dict[str, Any]:
        """Check if action is safe given battery level"""
        battery_threshold = 15.0  # Minimum safe battery level

        if state.battery_level < battery_threshold:
            high_power_actions = ['move_to', 'manipulation', 'dance']
            if action['action'] in high_power_actions:
                return {
                    'safe': False,
                    'reason': f'Insufficient battery ({state.battery_level}%) for power-intensive action',
                    'severity': 'medium'
                }

        return {'safe': True, 'reason': 'Battery level adequate', 'severity': 'low'}

    def _check_manipulation_safety(self, action: Dict[str, Any], state: RobotState) -> Dict[str, Any]:
        """Check safety of manipulation actions"""
        if action['action'] in ['pick_up', 'grasp', 'manipulate']:
            obj_name = action.get('parameters', {}).get('object', '')

            # Check for fragile objects
            fragile_objects = ['glass', 'ceramic', 'fragile', 'breakable']
            if any(fragile in obj_name.lower() for fragile in fragile_objects):
                return {
                    'safe': False,
                    'reason': f'Object {obj_name} may be fragile - requires careful handling',
                    'severity': 'medium'
                }

        return {'safe': True, 'reason': 'Manipulation appears safe', 'severity': 'low'}

    def _check_personal_space(self, action: Dict[str, Any], state: RobotState) -> Dict[str, Any]:
        """Check if action respects personal space"""
        if action['action'] == 'move_to':
            target_pos = action.get('parameters', {}).get('position', {})

            # Check if moving too close to detected humans
            for obj in state.detected_objects:
                if obj.get('type') == 'human':
                    human_pos = obj.get('position', {})
                    distance = self._calculate_distance(target_pos, human_pos)

                    if distance < 0.5:  # Too close to human
                        return {
                            'safe': False,
                            'reason': 'Action would violate human personal space',
                            'severity': 'high'
                        }

        return {'safe': True, 'reason': 'Respects personal space', 'severity': 'low'}

    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt(
            (pos1.get('x', 0) - pos2.get('x', 0))**2 +
            (pos1.get('y', 0) - pos2.get('y', 0))**2 +
            (pos1.get('z', 0) - pos2.get('z', 0))**2
        )

    def _check_consent(self, action: Dict[str, Any], state: RobotState) -> Dict[str, Any]:
        """Check if action has proper consent (simplified)"""
        social_actions = ['approach', 'greet', 'follow', 'assist']

        if action['action'] in social_actions:
            # In a real system, this would check for explicit consent
            # For now, assume consent is needed for social actions near humans
            humans_nearby = any(obj.get('type') == 'human' for obj in state.detected_objects)

            if humans_nearby:
                return {
                    'safe': True,  # Assume consent for example
                    'reason': 'Consent considerations noted',
                    'severity': 'medium'
                }

        return {'safe': True, 'reason': 'No consent issues', 'severity': 'low'}

    def check_action_safety(self, action: Dict[str, Any], state: RobotState) -> Dict[str, Any]:
        """Check if an action is safe to execute"""
        safety_results = []

        for rule in self.safety_rules:
            try:
                result = rule(action, state)
                safety_results.append(result)

                # If any rule indicates high severity unsafe, stop checking
                if not result['safe'] and result['severity'] == 'high':
                    return result
            except Exception as e:
                print(f"Error in safety rule: {e}")
                safety_results.append({
                    'safe': False,
                    'reason': f'Safety check error: {e}',
                    'severity': 'high'
                })

        # Aggregate results
        unsafe_results = [r for r in safety_results if not r['safe']]

        if unsafe_results:
            # Return the highest severity unsafe result
            highest_severity = max(unsafe_results, key=lambda x: ['low', 'medium', 'high'].index(x['severity']))
            return highest_severity
        else:
            return {'safe': True, 'reason': 'All safety checks passed', 'severity': 'low'}

class EthicalPlanner:
    def __init__(self, safety_checker: SafetyChecker, llm_planner: LLMPlanner):
        self.safety_checker = safety_checker
        self.llm_planner = llm_planner

        # Ethical principles
        self.ethical_principles = [
            "Do not harm humans",
            "Respect human autonomy and dignity",
            "Act transparently",
            "Protect privacy",
            "Be accountable for actions"
        ]

    async def plan_ethically(self, goal: str, current_state: RobotState) -> Optional[PlanningResponse]:
        """Plan with consideration of ethical principles"""
        # Create an ethical planning request
        ethical_context = {
            'ethical_principles': self.ethical_principles,
            'safety_constraints': True
        }

        request = PlanningRequest(
            goal=goal,
            current_state=current_state,
            context=ethical_context,
            constraints=[
                "Ensure all actions pass safety checks",
                "Respect human dignity and autonomy",
                "Maintain transparency in actions"
            ]
        )

        # Get plan from LLM
        response = await self.llm_planner.plan_task(request)

        if response:
            # Validate each action in the plan against safety rules
            validated_plan = []
            for action in response.plan:
                safety_check = self.safety_checker.check_action_safety(action, current_state)

                if safety_check['safe']:
                    validated_plan.append(action)
                else:
                    print(f"Action filtered for safety: {action} - {safety_check['reason']}")
                    # Try to find a safe alternative
                    alternative = await self._find_safe_alternative(action, current_state, safety_check)
                    if alternative:
                        validated_plan.extend(alternative)

            response.plan = validated_plan

        return response

    async def _find_safe_alternative(self, original_action: Dict[str, Any],
                                   current_state: RobotState,
                                   safety_issue: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Find a safe alternative to an unsafe action"""
        # This is a simplified implementation
        # In practice, you'd have a library of safe alternatives

        if original_action['action'] == 'move_to':
            # Try a safer path (simplified)
            params = original_action.get('parameters', {})
            original_target = params.get('position', {})

            # Create a safer alternative that maintains distance from humans
            safe_alternatives = []

            # Move to a position that's farther from humans
            for obj in current_state.detected_objects:
                if obj.get('type') == 'human':
                    human_pos = obj.get('position', {})
                    # Calculate a safe distance point
                    direction = np.array([original_target['x'] - human_pos['x'],
                                        original_target['y'] - human_pos['y']])
                    direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1, 0])

                    safe_pos = {
                        'x': human_pos['x'] + direction[0] * 1.0,  # 1m away
                        'y': human_pos['y'] + direction[1] * 1.0,
                        'z': original_target.get('z', 0)
                    }

                    safe_alternatives.append({
                        'action': 'move_to',
                        'parameters': {'position': safe_pos, 'safe_approach': True}
                    })

            return safe_alternatives or [original_action]  # Return original if no safe alternative found

        return None

# Example usage
async def safety_ethics_example():
    llm_planner = LLMPlanner(api_key="your-api-key")
    safety_checker = SafetyChecker()
    ethical_planner = EthicalPlanner(safety_checker, llm_planner)

    # Create a state that might trigger safety concerns
    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=25.0,  # Lower battery
        available_actions=["move_to", "pick_up", "speak"],
        current_task=None,
        detected_objects=[
            {"type": "human", "position": {"x": 0.3, "y": 0.2}, "name": "person1"},
            {"type": "object", "position": {"x": 1.0, "y": 0.5}, "name": "glass_cup"}
        ],
        human_interactions=[]
    )

    # Plan with ethical considerations
    response = await ethical_planner.plan_ethically("Move to the table and pick up the cup", state)

    if response:
        print(f"Ethical plan: {response.plan}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Confidence: {response.confidence}")
```

## Implementation Examples

### Complete Cognitive Architecture Example

```python
# complete_cognitive_architecture.py
import asyncio
from typing import Dict, Any, Optional
import logging

class HumanoidCognitiveSystem:
    def __init__(self, api_key: str):
        # Initialize components
        self.llm_planner = CachedLLMPlanner(api_key)
        self.memory_system = RobotMemory()
        self.context_manager = ContextManager(self.memory_system)
        self.perception_action_bridge = PerceptionActionBridge(
            self.llm_planner, self.memory_system
        )
        self.uncertainty_planner = UncertaintyAwarePlanner(self.llm_planner)
        self.safety_checker = SafetyChecker()
        self.ethical_planner = EthicalPlanner(self.safety_checker, self.llm_planner)

        # Real-time processing
        self.real_time_planner = RealTimeLLMPlanner(api_key)
        self.request_queue = PrioritizedRequestQueue()

        # Initialize perception-action bridge
        self._setup_perception_action_bridge()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_perception_action_bridge(self):
        """Set up the perception-action bridge with handlers and executors"""
        # Register perception handlers
        self.perception_action_bridge.register_perception_handler("camera", camera_handler)
        self.perception_action_bridge.register_perception_handler("lidar", lidar_handler)
        self.perception_action_bridge.register_perception_handler("microphone", microphone_handler)

        # Register action executors
        self.perception_action_bridge.register_action_executor("move_to", move_to_executor)
        self.perception_action_bridge.register_action_executor("pick_up", pick_up_executor)
        self.perception_action_bridge.register_action_executor("speak", speak_executor)

    async def process_command(self, command: str, robot_state: RobotState) -> bool:
        """Process a high-level command through the cognitive system"""
        self.logger.info(f"Processing command: {command}")

        # Update context with the new command
        self.context_manager.add_to_context("current_command", command)
        self.context_manager.add_to_context("command_timestamp", time.time())

        # Plan ethically with safety considerations
        response = await self.ethical_planner.plan_ethically(command, robot_state)

        if not response or not response.plan:
            self.logger.warning("No valid plan generated")
            return False

        # Execute the plan with uncertainty handling
        success = await self._execute_plan(response.plan, robot_state)

        # Update memory with the interaction
        self.memory_system.store_memory(
            memory_type="command_interaction",
            content={
                "command": command,
                "plan": response.plan,
                "success": success,
                "reasoning": response.reasoning
            },
            tags=["command", "interaction", "planning"],
            importance=response.confidence
        )

        return success

    async def _execute_plan(self, plan: List[Dict[str, Any]], robot_state: RobotState) -> bool:
        """Execute a plan with monitoring and adaptation"""
        for i, action in enumerate(plan):
            self.logger.info(f"Executing action {i+1}/{len(plan)}: {action['action']}")

            # Check safety before executing
            safety_check = self.safety_checker.check_action_safety(action, robot_state)
            if not safety_check['safe']:
                self.logger.warning(f"Action failed safety check: {safety_check['reason']}")
                continue  # Skip unsafe action

            # Execute the action
            success, confidence = await self.perception_action_bridge.execute_action(action)

            # Update belief state based on execution result
            self.uncertainty_planner.update_belief(
                {f"action_{action['action']}_success": success},
                confidence
            )

            # Record execution result for adaptation
            if hasattr(self, 'adaptive_planner'):
                self.adaptive_planner.record_execution_result(
                    action, success, confidence, random.uniform(1.0, 3.0)
                )

            if not success:
                self.logger.warning(f"Action {action['action']} failed")
                # In a real system, you might want to replan or try alternatives
                continue

        return True  # For simplicity, return True if we processed all actions

    async def run_cognitive_cycle(self, robot_state: RobotState):
        """Run a complete cognitive cycle"""
        # Process any queued requests
        try:
            request = await asyncio.wait_for(self.request_queue.get(), timeout=0.001)
            # Handle the request
            pass
        except asyncio.TimeoutError:
            pass  # No requests to process

        # Run perception-action cycle
        await self.perception_action_bridge.run_perception_action_cycle(robot_state)

        # Update context with current state
        self.context_manager.add_to_context("last_state_update", time.time())
        self.context_manager.add_to_context("current_battery", robot_state.battery_level)

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the cognitive system"""
        return {
            "components": {
                "llm_planner": "active",
                "memory_system": "active",
                "perception_action_bridge": "active",
                "safety_checker": "active"
            },
            "memory_stats": {
                "total_memories": len(self.memory_system.retrieve_memories(limit=1000)),
                "recent_accesses": "available"
            },
            "performance": self.real_time_planner.get_performance_metrics()
        }

# Example usage
async def cognitive_system_example():
    # Initialize the cognitive system
    cognitive_system = HumanoidCognitiveSystem(api_key="your-api-key")

    # Create initial robot state
    state = RobotState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=85.0,
        available_actions=["move_to", "pick_up", "speak"],
        current_task=None,
        detected_objects=[],
        human_interactions=[]
    )

    # Process some commands
    commands = [
        "Move to the kitchen",
        "Pick up the red cup",
        "Bring the cup to the table"
    ]

    for command in commands:
        success = await cognitive_system.process_command(command, state)
        print(f"Command '{command}' executed successfully: {success}")

        # Update state for next command (simplified)
        if "move" in command.lower():
            state.position = {"x": 2.0, "y": 1.0, "z": 0.0}
        elif "pick up" in command.lower():
            state.detected_objects = [{"name": "red_cup", "picked_up": True}]

    # Get system status
    status = cognitive_system.get_system_status()
    print(f"System status: {status}")

if __name__ == "__main__":
    asyncio.run(cognitive_system_example())
```

## Testing and Validation

### Cognitive System Tests

```python
# test_cognitive_system.py
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from cognitive_system import (
    LLMPlanner, RobotState, PlanningRequest, PlanningResponse,
    SafetyChecker, EthicalPlanner, RobotMemory
)

class TestLLMPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = LLMPlanner(api_key="test-key")

    @patch('openai.ChatCompletion.acreate')
    async def test_plan_task_success(self, mock_create):
        """Test successful task planning"""
        mock_response = AsyncMock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"plan": [{"action": "move_to", "parameters": {"target": "kitchen"}}], "reasoning": "Moving to kitchen", "confidence": 0.9, "estimated_time": 10.0}'

        mock_create.return_value = mock_response

        state = RobotState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=85.0,
            available_actions=["move_to", "pick_up"],
            current_task=None,
            detected_objects=[],
            human_interactions=[]
        )

        request = PlanningRequest(
            goal="Go to kitchen",
            current_state=state,
            context={},
            constraints=[]
        )

        response = await self.planner.plan_task(request)

        self.assertIsNotNone(response)
        self.assertEqual(len(response.plan), 1)
        self.assertEqual(response.plan[0]['action'], 'move_to')
        self.assertEqual(response.confidence, 0.9)

    @patch('openai.ChatCompletion.acreate')
    async def test_plan_task_failure(self, mock_create):
        """Test task planning failure"""
        mock_create.side_effect = Exception("API Error")

        state = RobotState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=85.0,
            available_actions=["move_to", "pick_up"],
            current_task=None,
            detected_objects=[],
            human_interactions=[]
        )

        request = PlanningRequest(
            goal="Go to kitchen",
            current_state=state,
            context={},
            constraints=[]
        )

        response = await self.planner.plan_task(request)

        self.assertIsNone(response)

class TestSafetyChecker(unittest.TestCase):
    def setUp(self):
        self.safety_checker = SafetyChecker()

    def test_collision_risk_detection(self):
        """Test collision risk detection"""
        action = {
            'action': 'move_to',
            'parameters': {
                'position': {'x': 2.0, 'y': 2.0, 'z': 0.0}
            }
        }

        state = RobotState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=85.0,
            available_actions=["move_to"],
            current_task=None,
            detected_objects=[{"name": "obstacle", "position": {"x": 1.0, "y": 1.0}}],
            human_interactions=[]
        )

        result = self.safety_checker.check_action_safety(action, state)

        # Should detect collision risk
        self.assertFalse(result['safe'])

    def test_battery_safety(self):
        """Test battery safety check"""
        action = {'action': 'move_to', 'parameters': {}}

        # Low battery state
        state = RobotState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=10.0,  # Very low
            available_actions=["move_to"],
            current_task=None,
            detected_objects=[],
            human_interactions=[]
        )

        result = self.safety_checker.check_action_safety(action, state)

        # Should flag low battery as unsafe for movement
        self.assertFalse(result['safe'])
        self.assertIn('battery', result['reason'].lower())

class TestRobotMemory(unittest.TestCase):
    def setUp(self):
        self.memory = RobotMemory()

    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving memories"""
        content = {"test": "data", "value": 42}
        memory_id = self.memory.store_memory("test_type", content, ["test", "data"])

        self.assertGreater(memory_id, 0)

        # Retrieve the memory
        memories = self.memory.retrieve_memories(memory_type="test_type", limit=1)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]['content']['test'], 'data')
        self.assertEqual(memories[0]['content']['value'], 42)

    def test_memory_retrieval_by_tags(self):
        """Test retrieving memories by tags"""
        # Store memories with different tags
        self.memory.store_memory("type1", {"data": 1}, ["tag1", "tag2"])
        self.memory.store_memory("type2", {"data": 2}, ["tag2", "tag3"])
        self.memory.store_memory("type3", {"data": 3}, ["tag1", "tag3"])

        # Retrieve by tag
        memories = self.memory.retrieve_memories(tags=["tag1"], limit=10)

        self.assertGreaterEqual(len(memories), 2)  # Should have at least 2 memories with tag1

class TestEthicalPlanner(unittest.TestCase):
    def setUp(self):
        self.mock_llm = Mock()
        self.mock_safety = Mock()
        self.ethical_planner = EthicalPlanner(self.mock_safety, self.mock_llm)

    @patch('asyncio.sleep', return_value=None)  # Mock sleep to avoid actual delays
    async def test_ethical_planning(self, mock_sleep):
        """Test ethical planning with safety validation"""
        # Mock LLM response
        mock_response = PlanningResponse(
            plan=[{"action": "move_to", "parameters": {"position": {"x": 1.0, "y": 1.0}}}],
            confidence=0.8,
            reasoning="Moving to position",
            estimated_time=5.0
        )
        self.mock_llm.plan_task = AsyncMock(return_value=mock_response)

        # Mock safety check
        self.mock_safety.check_action_safety = Mock(return_value={'safe': True, 'reason': 'OK', 'severity': 'low'})

        state = RobotState(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            battery_level=85.0,
            available_actions=["move_to"],
            current_task=None,
            detected_objects=[],
            human_interactions=[]
        )

        response = await self.ethical_planner.plan_ethically("Go somewhere", state)

        self.assertIsNotNone(response)
        self.assertEqual(len(response.plan), 1)

def run_cognitive_tests():
    """Run all cognitive system tests"""
    # Run tests in an async context
    async def run_tests():
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__('__main__', globals(), locals(), ['']))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    asyncio.run(run_tests())

if __name__ == '__main__':
    run_cognitive_tests()
```

## Summary and Next Steps

### Key Takeaways

1. **LLM Integration**: Large Language Models can significantly enhance robotic cognitive capabilities by providing natural language understanding and reasoning.

2. **Hierarchical Planning**: Breaking down complex goals into manageable subtasks improves planning effectiveness and execution success.

3. **Context Awareness**: Maintaining and utilizing context information enables more intelligent and adaptive robot behavior.

4. **Uncertainty Handling**: Real-world robotics involves uncertainty, requiring probabilistic reasoning and adaptive planning.

5. **Safety First**: Safety checks and ethical considerations must be integrated into all planning processes.

6. **Real-time Constraints**: Optimizing LLM usage for real-time applications requires careful consideration of latency and resource usage.

### Next Steps

1. **Deployment**: Deploy the cognitive system on actual humanoid hardware for real-world testing.

2. **Fine-tuning**: Fine-tune LLMs on robotics-specific datasets for improved performance.

3. **Multi-modal Integration**: Integrate vision, audio, and other sensory modalities with LLM planning.

4. **Learning Systems**: Implement learning from interaction to improve planning over time.

5. **Edge Computing**: Optimize for edge deployment to reduce latency and improve privacy.

6. **Collaborative Robots**: Extend to multi-robot coordination and human-robot collaboration.

### Advanced Topics

For further development, consider:
- Federated learning for privacy-preserving improvement
- Reinforcement learning integration with LLM planning
- Multi-agent systems with distributed cognitive architectures
- Explainable AI for transparent decision-making
- Continuous learning and adaptation systems

## Exercises

1. Implement a custom safety checker for your specific robot platform
2. Create a memory system that uses vector embeddings for semantic search
3. Develop an uncertainty model specific to your robot's capabilities
4. Build a real-time performance monitoring dashboard
5. Create a simulation environment to test cognitive planning without hardware

## References

- OpenAI API: https://platform.openai.com/docs/api-reference
- ROS 2 Documentation: https://docs.ros.org/en/humble/
- Cognitive Robotics: https://www.cognitiverobotics.org/
- Planning Algorithms: http://planning.cs.uiuc.edu/
- Safety in Robotics: https://www.ieee-ras.org/standards-committee