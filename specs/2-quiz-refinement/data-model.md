# Data Model: Quiz System Refinement

**Feature**: 2-quiz-refinement
**Created**: 2025-12-15

## Entity Definitions

### Module
**Description**: Educational content unit covering specific topics in Physical AI & Humanoid Robotics

**Attributes**:
- `id` (string): Unique identifier for the module (e.g., "module1", "module2")
- `title` (string): Display title of the module
- `content` (string): Main instructional content of the module
- `authoritativeQuiz` (AuthoritativeQuiz): Associated full quiz from quizzes/ directory
- `inlineKnowledgeChecks` (Array<InlineKnowledgeCheck>): Brief self-checks within module content

**Relationships**:
- One-to-One with AuthoritativeQuiz (each module has exactly one authoritative quiz)
- One-to-Many with InlineKnowledgeCheck (module can have multiple inline checks)

### AuthoritativeQuiz
**Description**: Full assessment from quizzes/ directory that serves as the authoritative source

**Attributes**:
- `id` (string): Unique identifier, typically matches module ID
- `title` (string): Display title of the quiz
- `module` (string): Reference to the parent module
- `questions` (Array<Question>): Collection of questions from the authoritative source
- `sourcePath` (string): File path to the quiz in the quizzes/ directory
- `instructions` (string): General instructions for taking the quiz

**Relationships**:
- Many-to-One with Module (many quizzes belong to modules)
- One-to-Many with Question (one quiz contains multiple questions)

### InlineKnowledgeCheck
**Description**: Brief self-check embedded within module content for immediate reinforcement

**Attributes**:
- `id` (string): Unique identifier within the module
- `moduleId` (string): Reference to the parent module
- `position` (number): Position within the module content (sequence number)
- `question` (string): The question text for the knowledge check
- `options` (Array<string>): Available answer choices
- `correctAnswer` (number): Index of the correct answer option
- `explanation` (string): Brief explanation of the correct answer
- `quizLink` (string): Link to the full module quiz for comprehensive assessment

**Relationships**:
- Many-to-One with Module (many inline checks belong to one module)

### Question
**Description**: Individual assessment item from the authoritative quiz source

**Attributes**:
- `id` (string): Unique identifier within the quiz
- `type` (string): Type of question (currently "multiple-choice" for authoritative quizzes)
- `text` (string): The question text
- `options` (Array<string>): Available answer choices (A, B, C, D format)
- `correctAnswer` (string): The correct answer option (A, B, C, or D)
- `explanation` (string): Explanation of the correct answer (derived from context)
- `difficulty` (string): Difficulty level ("easy", "medium", "hard")

**Relationships**:
- Many-to-One with AuthoritativeQuiz (many questions belong to one quiz)

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

### AuthoritativeQuiz State Model
```
Initial (Not Started)
    ↓ (User begins quiz)
In Progress
    ↓ (User answers questions)
Partially Completed
    ↓ (User completes all questions)
Completed
    ↓ (User requests retake)
Reset to Initial
```

### InlineKnowledgeCheck State Model
```
Ready for Response
    ↓ (User selects answer)
Response Submitted
        ├── ↓ (User is correct)
        └── Displays positive feedback
        ├── ↓ (User is incorrect)
        └── Displays explanation and feedback
        ↓ (User continues learning)
Ready for Next Content
```

## Validation Rules

### Module Validation
- `id` must be unique across all modules
- `title` must not be empty
- `authoritativeQuiz` must be present and reference a valid quiz file

### AuthoritativeQuiz Validation
- `sourcePath` must reference an existing file in the quizzes/ directory
- `questions` array must contain 3 or more questions
- `title` must match the title in the source file
- Content must not be modified from the authoritative source

### InlineKnowledgeCheck Validation
- `moduleId` must reference an existing module
- `question` must not be empty
- `options` array must contain 2-4 options
- `correctAnswer` must be a valid index in the options array
- `explanation` must be provided for the correct answer
- Maximum of 5 inline checks per module to maintain learning flow

### Question Validation (from authoritative source)
- `text` must not be empty
- `options` array must contain exactly 4 options (A, B, C, D format)
- `correctAnswer` must be a valid option (A, B, C, or D)
- Question must exist in the corresponding source file in quizzes/ directory

### UserResponse Validation
- `questionId` must reference an existing question in the current assessment
- `selectedOption` must be a valid option for the question
- Responses are temporary and tied to session lifecycle

## Constraints

### Content Integrity Constraints
- Quiz content from `quizzes/` directory must not be modified or duplicated
- All authoritative quiz questions must match exactly those in the source files
- Inline knowledge checks must not duplicate content from full module quizzes
- Changes to source quiz files should be reflected automatically in the system

### Navigation Constraints
- Users must be able to access the full quiz directly from inline knowledge checks
- Full quizzes must appear after all instructional content and next-step sections
- Inline checks should not interrupt the primary learning flow
- Relationship between module content and quizzes must be explicit

### Assessment Model Constraints
- Inline knowledge checks must remain concise (1-2 questions maximum)
- Inline checks are supplementary and must not replace full module quizzes
- Full quizzes contain comprehensive assessment content
- Both assessment types must be clearly distinguished from each other

### Display Constraints
- Inline knowledge checks must have distinct visual styling from full quizzes
- Full quizzes must be clearly marked as the authoritative assessment
- Links between inline checks and full quizzes must be prominent
- Assessment sections must be clearly separated from instructional content