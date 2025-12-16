# Quickstart Guide: Module-Level Assessment Implementation

**Feature**: 1-module-assessments
**Created**: 2025-12-15

## Overview
This guide provides a quick setup for implementing quiz assessments at the end of each module in the Physical AI & Humanoid Robotics textbook. The implementation uses embedded quiz sections within each module's Markdown file using MDX components.

## Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager
- Docusaurus CLI installed (`npm install -g @docusaurus/cli`)
- Access to module content files in the `docs/` directory

## Setup Steps

### 1. Create Quiz Components
Create a new directory for components and add the quiz components:

```bash
mkdir -p src/components/quiz
```

Create `src/components/quiz/QuizSection.jsx`:
```jsx
import React, { useState } from 'react';
import QuestionCard from './QuestionCard';

const QuizSection = ({ moduleId, title = "Knowledge Check", questions = [] }) => {
  const [responses, setResponses] = useState({});
  const [submitted, setSubmitted] = useState(false);

  const handleAnswerSelect = (questionId, optionIndex) => {
    if (!submitted) {
      setResponses(prev => ({
        ...prev,
        [questionId]: optionIndex
      }));
    }
  };

  const handleSubmit = () => {
    setSubmitted(true);
  };

  const resetQuiz = () => {
    setResponses({});
    setSubmitted(false);
  };

  const correctCount = questions.filter(q =>
    responses[q.id] === q.correctAnswer && submitted
  ).length;

  return (
    <div className="quiz-section">
      <h3>{title}</h3>
      <div className="quiz-instructions">
        <p>Test your understanding of the key concepts from this module.</p>
      </div>

      {questions.map((question, index) => (
        <QuestionCard
          key={question.id}
          question={question}
          selectedOption={responses[question.id]}
          onAnswerSelect={(optionIndex) => handleAnswerSelect(question.id, optionIndex)}
          submitted={submitted}
        />
      ))}

      {!submitted ? (
        <button onClick={handleSubmit} className="quiz-submit-btn">
          Submit Answers
        </button>
      ) : (
        <div className="quiz-results">
          <h4>Results: {correctCount} of {questions.length} correct</h4>
          <button onClick={resetQuiz} className="quiz-reset-btn">
            Retake Quiz
          </button>
        </div>
      )}
    </div>
  );
};

export default QuizSection;
```

Create `src/components/quiz/QuestionCard.jsx`:
```jsx
import React from 'react';

const QuestionCard = ({ question, selectedOption, onAnswerSelect, submitted }) => {
  const isCorrect = submitted && selectedOption === question.correctAnswer;
  const isSelected = selectedOption !== undefined;
  const userWasCorrect = isSelected && isCorrect;

  return (
    <div className={`question-card ${submitted && isCorrect ? 'correct' : ''} ${submitted && !isCorrect ? 'incorrect' : ''}`}>
      <h5>Question {question.id.replace('q', '')}: {question.text}</h5>
      <ul className="question-options">
        {question.options.map((option, index) => {
          const isOptionCorrect = index === question.correctAnswer;
          const isOptionSelected = index === selectedOption;

          let optionClass = 'option';
          if (submitted) {
            if (isOptionCorrect) optionClass += ' correct-option';
            if (isOptionSelected && !isOptionCorrect) optionClass += ' incorrect-selected';
          }

          return (
            <li
              key={index}
              className={optionClass}
              onClick={() => !submitted && onAnswerSelect(index)}
            >
              <span className="option-letter">{String.fromCharCode(65 + index)}.</span>
              <span className="option-text">{option}</span>
              {submitted && isOptionSelected && (
                <span className="option-feedback">
                  {isOptionCorrect ? ' ✓' : ' ✗'}
                </span>
              )}
            </li>
          );
        })}
      </ul>

      {submitted && userWasCorrect && (
        <div className="explanation correct-explanation">
          <strong>Correct!</strong> {question.explanation}
        </div>
      )}

      {submitted && isSelected && !userWasCorrect && (
        <div className="explanation incorrect-explanation">
          <strong>Incorrect.</strong> {question.explanation}
        </div>
      )}
    </div>
  );
};

export default QuestionCard;
```

### 2. Add CSS Styling
Add the following CSS to your `src/css/custom.css` file:

```css
.quiz-section {
  margin-top: 2rem;
  padding: 1.5rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  background-color: #f9f9f9;
}

.quiz-section h3 {
  color: #2c3e50;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.5rem;
  margin-top: 0;
}

.quiz-instructions {
  margin-bottom: 1.5rem;
  font-style: italic;
  color: #666;
}

.question-card {
  margin-bottom: 1.5rem;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
}

.question-card.correct {
  border-color: #28a745;
  background-color: #f8fff9;
}

.question-card.incorrect {
  border-color: #dc3545;
  background-color: #fff8f8;
}

.question-options {
  list-style: none;
  padding: 0;
}

.option {
  padding: 0.75rem;
  margin: 0.25rem 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.option:hover {
  background-color: #f5f5f5;
}

.option.correct-option {
  background-color: #d4edda;
  border-color: #28a745;
}

.option.incorrect-selected {
  background-color: #f8d7da;
  border-color: #dc3545;
}

.option-letter {
  font-weight: bold;
  margin-right: 0.5rem;
}

.option-feedback {
  float: right;
  font-weight: bold;
}

.explanation {
  margin-top: 0.5rem;
  padding: 0.75rem;
  border-radius: 4px;
  font-size: 0.9rem;
}

.correct-explanation {
  background-color: #d4edda;
  border: 1px solid #c3e6cb;
  color: #155724;
}

.incorrect-explanation {
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  color: #721c24;
}

.quiz-submit-btn, .quiz-reset-btn {
  padding: 0.5rem 1rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.quiz-submit-btn:hover, .quiz-reset-btn:hover {
  background-color: #0056b3;
}
```

### 3. Add Quiz Data Structure
Create a quiz data file for each module in `src/data/quizzes/`:

```bash
mkdir -p src/data/quizzes
```

Create `src/data/quizzes/module1.json` as an example:
```json
{
  "moduleId": "module1",
  "title": "Knowledge Check",
  "questions": [
    {
      "id": "q1",
      "type": "multiple-choice",
      "text": "What is the primary purpose of ROS 2 in robotics development?",
      "options": [
        "To provide a hardware platform for robots",
        "To serve as a communication framework between robot components",
        "To replace all existing robot software",
        "To act as a robot simulation environment"
      ],
      "correctAnswer": 1,
      "explanation": "ROS 2 (Robot Operating System 2) serves as a middleware framework that provides services such as hardware abstraction, device drivers, libraries, and message-passing between different robot components."
    },
    {
      "id": "q2",
      "type": "multiple-choice",
      "text": "Which of the following is a key feature of ROS 2 compared to ROS 1?",
      "options": [
        "Python-only programming interface",
        "Single-threaded execution model",
        "DDS-based middleware for communication",
        "Limited package management"
      ],
      "correctAnswer": 2,
      "explanation": "ROS 2 uses DDS (Data Distribution Service) as its underlying middleware, providing better real-time performance, fault tolerance, and language independence compared to ROS 1's custom TCPROS/UDPROS implementations."
    }
  ]
}
```

### 4. Integrate Quiz into Module
Add the quiz to the end of a module file (e.g., `docs/module1/index.md`):

```markdown
---
sidebar_position: 1
---

# Module 1: ROS 2 Fundamentals

[Your existing module content here...]

<!-- Quiz Section -->
import QuizSection from '@site/src/components/quiz/QuizSection';
import module1QuizData from '@site/src/data/quizzes/module1.json';

<QuizSection
  moduleId={module1QuizData.moduleId}
  title={module1QuizData.title}
  questions={module1QuizData.questions}
/>
```

### 5. Update Navigation
Ensure the quiz section is accessible by updating any internal navigation or table of contents within the module if needed.

## Testing the Setup

1. Start the development server:
```bash
npm run start
```

2. Navigate to a module page with a quiz
3. Verify that:
   - The quiz section appears at the end of the module
   - Questions are properly displayed
   - Answer selection works
   - Feedback is shown after submission
   - Results are calculated correctly

## Deployment
The quiz functionality will be built into the static site during the regular Docusaurus build process:
```bash
npm run build
```

The quiz components will be bundled with the rest of the site and deployed as part of the standard static site deployment process.