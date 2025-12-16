# Contract: Quiz Data Structure

## Purpose
Defines the expected data structure for quiz content used by the AuthoritativeQuizComponent in dedicated quiz pages.

## Version
1.0

## Data Structure

### Quiz Object
```json
{
  "id": "string",
  "title": "string",
  "module": "string",
  "questions": [
    {
      "id": "string",
      "text": "string",
      "options": ["string"],
      "correctAnswer": "number",
      "explanation": "string"
    }
  ]
}
```

### Field Definitions
- **id**: Unique identifier for the quiz (e.g., "module1")
- **title**: Display title for the quiz (e.g., "Module 1 Quiz - ROS 2 Fundamentals")
- **module**: Module identifier (e.g., "module1")
- **questions**: Array of question objects
  - **id**: Unique identifier for the question (e.g., "q1")
  - **text**: The question text to display
  - **options**: Array of answer option strings
  - **correctAnswer**: Zero-based index of the correct option in the options array
  - **explanation**: Explanation text for the correct answer

## Validation Rules
- All fields are required
- questions array must contain at least one question
- correctAnswer must be a valid index within the options array (0 <= correctAnswer < options.length)
- options array must contain at least one option

## Source Mapping
- Data is sourced from /quizzes/moduleX-quiz.md files
- Processed through build-time parser to generate JSON format
- Maintains exact content from source files without modification

## Compatibility
- Compatible with AuthoritativeQuizComponent version 1.0+
- Backward compatible with existing quiz markdown format
- Forward compatible with additional optional fields