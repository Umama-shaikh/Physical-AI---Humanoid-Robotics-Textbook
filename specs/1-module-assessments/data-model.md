# Data Model: Module-Level Assessment Requirements

**Feature**: 1-module-assessments
**Created**: 2025-12-15

## Entity Definitions

### Module
**Description**: Educational content unit covering specific topics in Physical AI & Humanoid Robotics

**Attributes**:
- `id` (string): Unique identifier for the module (e.g., "module1", "module2")
- `title` (string): Display title of the module
- `content` (string): Main instructional content of the module
- `assessment` (Assessment): Associated quiz/assessment for the module

**Relationships**:
- One-to-One with Assessment (each module has exactly one assessment)

### Assessment
**Description**: Collection of questions that test understanding of module content

**Attributes**:
- `id` (string): Unique identifier, typically derived from parent module ID
- `title` (string): Display title (e.g., "Knowledge Check", "Module Quiz")
- `moduleId` (string): Reference to the parent module
- `questions` (Array<Question>): Collection of questions in the assessment
- `instructions` (string): Instructions for completing the assessment

**Relationships**:
- One-to-Many with Question (one assessment contains multiple questions)
- Many-to-One with Module (many assessments belong to modules)

### Question
**Description**: Individual assessment item that tests understanding of specific module concepts

**Attributes**:
- `id` (string): Unique identifier within the assessment
- `type` (string): Type of question (e.g., "multiple-choice", "true-false")
- `text` (string): The question text
- `options` (Array<string>): Available answer choices (for multiple choice)
- `correctAnswer` (number | string): Index of correct option or the correct answer text
- `explanation` (string): Explanation of the correct answer
- `difficulty` (string): Difficulty level ("easy", "medium", "hard")

**Relationships**:
- Many-to-One with Assessment (many questions belong to one assessment)

### UserResponse (Session-based)
**Description**: Temporary tracking of user's response to a question during a session

**Attributes**:
- `questionId` (string): Reference to the question being answered
- `selectedOption` (number | string): User's selected answer
- `isCorrect` (boolean): Whether the answer was correct
- `timestamp` (Date): When the response was recorded
- `sessionId` (string): Session identifier to group responses

**Relationships**:
- Many-to-One with Question (many responses can be recorded for questions over time)

## State Transitions

### Assessment State Model
```
Initial (Not Started)
    ↓ (User begins quiz)
In Progress
    ↓ (User answers questions)
Partially Completed
    ↓ (User completes all questions)
Completed
```

### Question State Model
```
Unanswered
    ↓ (User selects answer)
Answered
        ├── ↓ (User changes answer)
        └── (Remains answered until reset)
```

## Validation Rules

### Module Validation
- `id` must be unique across all modules
- `title` must not be empty
- `assessment` must be present and valid

### Assessment Validation
- `moduleId` must reference an existing module
- `questions` array must contain 3-10 questions
- `title` must be clearly labeled as an assessment ("Quiz", "Knowledge Check", etc.)

### Question Validation
- `text` must not be empty
- For multiple-choice: `options` array must contain 2-6 options
- `correctAnswer` must be a valid index or value in the available options
- `explanation` must be provided for the correct answer

### UserResponse Validation
- `questionId` must reference an existing question in the current assessment
- `selectedOption` must be a valid option for the question
- Responses are temporary and tied to session lifecycle

## Constraints

### Content Constraints
- Each module must have exactly one associated assessment
- Assessment questions must directly relate to the module's content
- Assessment sections must appear at the end of module content
- Assessment content must not duplicate instructional content

### Navigation Constraints
- Users must be able to access the assessment directly from module content
- Navigation between module content and assessment must be seamless
- Assessment must be clearly distinguished from main content

### Display Constraints
- Assessment sections must have distinct visual styling
- Assessment titles must clearly indicate the purpose
- Correct answers and explanations must be provided after user response