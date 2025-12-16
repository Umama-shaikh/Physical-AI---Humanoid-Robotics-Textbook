# Quiz Authoring Guidelines

## Purpose
These guidelines help content creators develop effective quiz questions that reinforce key concepts and enhance the learning experience in the Physical AI & Humanoid Robotics textbook.

## Quiz Structure

### Question Format
- Use multiple-choice format with 3-6 options
- Include exactly one correct answer
- Provide clear explanations for both correct and incorrect answers
- Keep questions focused on one concept at a time

### Content Alignment
- Each question should directly relate to a key concept from the module
- Questions should test understanding, not just memorization
- Avoid trick questions or overly complex scenarios
- Ensure questions cover different difficulty levels (easy, medium, hard)

## Writing Effective Questions

### Question Text
- Write clear, concise questions that are easy to understand
- Use the same terminology as the module content
- Avoid double negatives or complex sentence structures
- Make sure the question stands alone without needing context from the module

### Answer Options
- Keep all options similar in length and complexity
- Use plausible distractors that reflect common misconceptions
- Avoid "all of the above" or "none of the above" options
- Make sure incorrect options are clearly wrong, not just slightly wrong

### Explanations
- Provide detailed explanations for the correct answer
- Explain why incorrect options are wrong
- Reference specific concepts from the module when relevant
- Keep explanations educational and informative

## Technical Requirements

### JSON Structure
Quiz data must follow this structure in the appropriate module file:

```json
{
  "id": "q1",  // Unique identifier for the question
  "type": "multiple-choice",  // Currently only multiple-choice is supported
  "text": "Question text here",  // The actual question
  "options": [  // Array of answer options
    "Option A",
    "Option B",
    "Option C",
    "Option D"
  ],
  "correctAnswer": 1,  // Index of the correct answer (0-based)
  "explanation": "Explanation of the correct answer",  // Detailed explanation
  "difficulty": "medium"  // Difficulty level: "easy", "medium", or "hard"
}
```

### Validation
- All fields are required
- Options array must have 2-6 options
- correctAnswer must be a valid index for the options array
- Question text and explanation must not be empty

## Best Practices

### Difficulty Distribution
- Include a mix of difficulty levels within each quiz
- Start with easier questions to build confidence
- Gradually increase difficulty
- Aim for 40% easy, 40% medium, 20% hard

### Learning Objectives
- Align questions with the module's learning objectives
- Test both conceptual understanding and practical application
- Include questions that connect multiple concepts when appropriate
- Ensure questions reflect the most important content from the module

### Review Process
- Have another subject matter expert review questions
- Test questions with a small group of learners if possible
- Update questions based on learner feedback
- Regularly review and update explanations

## Examples

### Good Question:
"What does DDS stand for in the context of ROS 2?"
- Tests specific terminology from the module
- Clear and concise
- Has unambiguous correct answer

### Avoid:
"Which of the following is true about ROS 2? (select all that apply)"
- Multiple correct answers are not supported
- "Select all that apply" format not available

## Quality Assurance
- Test all quiz functionality after adding new questions
- Verify that explanations are accurate and helpful
- Ensure questions are free from typos or errors
- Confirm that difficulty level is appropriate for the target audience