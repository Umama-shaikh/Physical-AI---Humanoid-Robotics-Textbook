# Quiz System Documentation

## Overview
The Physical AI & Humanoid Robotics textbook includes interactive quiz assessments at the end of each module to reinforce key concepts and provide self-assessment opportunities for learners.

## Features
- Multiple-choice questions with immediate feedback
- Visual distinction from instructional content
- Progress tracking within each quiz session
- Ability to retake quizzes for reinforcement

## Technical Implementation
The quiz system is built using React components integrated into the Docusaurus documentation framework:

- **QuizSection.jsx**: Main container component that manages quiz state and displays results
- **QuestionCard.jsx**: Individual question component with answer selection and feedback
- **quizValidation.js**: Validation functions for quiz data integrity
- **CSS styling**: Custom styles in `src/css/custom.css` for visual distinction

## How to Use
Learners can access the quiz at the end of each module by scrolling to the "Knowledge Check" section. The quiz includes:

1. Multiple-choice questions that reinforce key concepts from the module
2. Immediate feedback when answers are submitted
3. Ability to retake the quiz to reinforce learning
4. Results summary showing the percentage of correct answers

## For Content Creators
To add or modify quiz questions for a module:

1. Update the corresponding JSON file in `src/data/quizzes/`
2. Follow the established data structure with id, text, options, correctAnswer, and explanation
3. Ensure questions align with the module's learning objectives
4. Test the quiz functionality after making changes

## Accessibility
The quiz system includes accessibility features:
- Keyboard navigation support (Enter/Space to select options)
- ARIA labels and roles for screen readers
- Proper focus management
- Semantic HTML structure