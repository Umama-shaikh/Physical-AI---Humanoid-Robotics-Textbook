# Data Model: Physical AI & Humanoid Robotics Book

## Book Module
- **Description**: A self-contained learning unit covering specific technologies (ROS 2, simulation, AI, VLA)
- **Attributes**:
  - module_id: string (unique identifier like "module1")
  - title: string (display title)
  - description: string (brief overview)
  - learning_objectives: array of strings (specific goals for the module)
  - chapters: array of Chapter references
  - duration: integer (estimated hours to complete)
- **Relationships**: Contains multiple Chapters; referenced by Book

## Chapter
- **Description**: An individual lesson within a module with specific learning objectives and examples
- **Attributes**:
  - chapter_id: string (unique identifier like "module1-chapter1")
  - title: string (display title)
  - content: string (Markdown content)
  - learning_objectives: array of strings
  - prerequisites: array of string references to other chapters
  - examples: array of Example Code references
  - exercises: array of Exercise references
  - estimated_reading_time: integer (in minutes)
- **Relationships**: Belongs to one Module; contains multiple Example Code and Exercises

## Example Code
- **Description**: Runnable Python, URDF, SDF, and configuration files that demonstrate concepts
- **Attributes**:
  - example_id: string (unique identifier)
  - title: string (display title)
  - file_path: string (relative path to file)
  - file_type: string (python, urdf, sdf, config, etc.)
  - description: string (what the example demonstrates)
  - prerequisites: array of string references to concepts/chapters
  - dependencies: array of strings (required packages/tools)
- **Relationships**: Belongs to one or more Chapters

## Assessment Quiz
- **Description**: Module-specific quiz to validate student understanding
- **Attributes**:
  - quiz_id: string (unique identifier like "module1-quiz")
  - module_id: string (reference to parent module)
  - title: string (display title)
  - questions: array of Question objects
  - passing_score: integer (percentage required to pass)
  - time_limit: integer (minutes, 0 if untimed)
- **Relationships**: Belongs to one Module; contains multiple Questions

## Question
- **Description**: Individual assessment item within a quiz
- **Attributes**:
  - question_id: string (unique identifier)
  - quiz_id: string (reference to parent quiz)
  - question_text: string (the actual question)
  - question_type: string (multiple-choice, true-false, short-answer)
  - options: array of string (for multiple choice)
  - correct_answer: string (the correct response)
  - explanation: string (why this is correct)
  - difficulty: string (beginner, intermediate, advanced)
- **Relationships**: Belongs to one Assessment Quiz

## Visual Aid
- **Description**: Diagrams, illustrations, and images that support text-based explanations
- **Attributes**:
  - visual_id: string (unique identifier)
  - title: string (description of the visual)
  - file_path: string (relative path to image file)
  - file_type: string (png, svg, jpg, etc.)
  - caption: string (explanatory text)
  - alt_text: string (accessibility description)
  - related_concepts: array of strings (concepts this visual explains)
- **Relationships**: Referenced by multiple Chapters

## Learning Path
- **Description**: Sequence of modules and chapters that defines the recommended learning journey
- **Attributes**:
  - path_id: string (unique identifier)
  - title: string (display title like "Complete Book Path")
  - description: string (overview of the path)
  - modules_order: array of Module references (ordered sequence)
  - estimated_duration: integer (total hours for complete path)
- **Relationships**: Contains multiple Modules in a specific order