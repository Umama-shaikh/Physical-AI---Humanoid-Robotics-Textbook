# Quickstart Guide: Quiz System Refinement Implementation

**Feature**: 2-quiz-refinement
**Created**: 2025-12-15

## Overview
This guide provides a quick setup for implementing the refined quiz system that distinguishes between inline knowledge checks and authoritative end-of-module quizzes. The implementation reuses existing quiz content from the `quizzes/` directory while adding inline assessment capabilities.

## Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager
- Docusaurus CLI installed (`npm install -g @docusaurus/cli`)
- Access to module content files in the `docs/` directory
- Access to existing quiz files in the `quizzes/` directory

## Setup Steps

### 1. Create Quiz Components
Create a new directory for refined quiz components and add the necessary files:

```bash
mkdir -p src/components/quiz-refined
```

Create `src/components/quiz-refined/AuthoritativeQuizComponent.jsx`:
```jsx
import React, { useState, useEffect } from 'react';
import fs from 'fs';
import path from 'path';

// This component loads and renders the authoritative quiz from the quizzes/ directory
const AuthoritativeQuizComponent = ({ moduleId, title = "Module Quiz", quizSourcePath }) => {
  const [questions, setQuestions] = useState([]);
  const [responses, setResponses] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(true);

  // In a real implementation, this would load the quiz file content
  // For now, we'll use a placeholder that would be populated with parsed content
  useEffect(() => {
    // This would be replaced with actual file loading logic during build time
    // Since we can't load files client-side in this context, this is conceptual
    const loadQuiz = async () => {
      try {
        // Placeholder for quiz loading logic
        // In practice: fetch quiz content from quizSourcePath and parse it
        const mockQuestions = [
          {
            id: 'q1',
            text: 'Sample question from authoritative quiz',
            options: ['Option A', 'Option B', 'Option C', 'Option D'],
            correctAnswer: 1,
            explanation: 'Explanation for the correct answer'
          }
        ];
        setQuestions(mockQuestions);
        setLoading(false);
      } catch (error) {
        console.error('Error loading quiz:', error);
        setLoading(false);
      }
    };

    loadQuiz();
  }, [quizSourcePath]);

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

  if (loading) {
    return <div className="quiz-loading">Loading quiz...</div>;
  }

  const correctCount = questions.filter(q =>
    responses[q.id] === q.correctAnswer && submitted
  ).length;

  return (
    <div className="authoritative-quiz-section" role="region" aria-labelledby="quiz-title">
      <h3 id="quiz-title">{title}</h3>
      <div className="quiz-instructions" id="quiz-instructions">
        <p>Complete assessment covering all concepts from this module.</p>
      </div>

      {questions.map((question, index) => {
        const selectedOption = responses[question.id];
        const isCorrect = submitted && selectedOption === question.correctAnswer;
        const isSelected = selectedOption !== undefined;
        const userWasCorrect = isSelected && isCorrect;

        return (
          <div
            key={question.id}
            className={`question-card ${submitted && isCorrect ? 'correct' : ''} ${submitted && !isCorrect ? 'incorrect' : ''}`}
            role="group"
            aria-labelledby={`question-${question.id}`}
          >
            <h5 id={`question-${question.id}`}>Question {index + 1}: {question.text}</h5>
            <ul className="question-options" role="radiogroup" aria-describedby="quiz-instructions">
              {question.options.map((option, optIndex) => {
                const optionId = `question-${question.id}-option-${optIndex}`;
                const isOptionSelected = optIndex === selectedOption;
                const isOptionCorrect = optIndex === question.correctAnswer;

                let optionClass = 'option';
                if (submitted) {
                  if (isOptionCorrect) optionClass += ' correct-option';
                  if (isOptionSelected && !isOptionCorrect) optionClass += ' incorrect-selected';
                }

                return (
                  <li
                    key={optIndex}
                    id={optionId}
                    className={optionClass}
                    onClick={() => !submitted && handleAnswerSelect(question.id, optIndex)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        !submitted && handleAnswerSelect(question.id, optIndex);
                      }
                    }}
                    role="radio"
                    aria-checked={isOptionSelected ? 'true' : 'false'}
                    tabIndex={0}
                    aria-describedby={`question-${question.id}`}
                  >
                    <span className="option-letter" aria-hidden="true">{String.fromCharCode(65 + optIndex)}.</span>
                    <span className="option-text">{option}</span>
                    {submitted && isOptionSelected && (
                      <span className="option-feedback" aria-label={isOptionCorrect ? 'Correct' : 'Incorrect'}>
                        {isOptionCorrect ? ' ✓' : ' ✗'}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>

            {submitted && userWasCorrect && (
              <div className="explanation correct-explanation" role="status">
                <strong>Correct!</strong> {question.explanation}
              </div>
            )}

            {submitted && isSelected && !userWasCorrect && (
              <div className="explanation incorrect-explanation" role="status">
                <strong>Incorrect.</strong> {question.explanation}
              </div>
            )}
          </div>
        );
      })}

      {!submitted ? (
        <button
          onClick={handleSubmit}
          className="quiz-submit-btn"
          aria-describedby="quiz-instructions"
        >
          Submit Answers
        </button>
      ) : (
        <div className="quiz-results" role="status" aria-live="polite">
          <h4>Results: {correctCount} of {questions.length} correct</h4>
          <button
            onClick={resetQuiz}
            className="quiz-reset-btn"
            aria-label="Retake quiz"
          >
            Retake Quiz
          </button>
        </div>
      )}
    </div>
  );
};

export default AuthoritativeQuizComponent;
```

Create `src/components/quiz-refined/InlineKnowledgeCheck.jsx`:
```jsx
import React, { useState } from 'react';

// This component provides a brief inline knowledge check
const InlineKnowledgeCheck = ({
  question,
  options,
  correctAnswer,
  explanation,
  quizLink = "#module-quiz",
  position = 1
}) => {
  const [selectedOption, setSelectedOption] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleSelect = (optionIndex) => {
    if (!submitted) {
      setSelectedOption(optionIndex);
      const isCorrect = optionIndex === correctAnswer;
      setShowExplanation(true);

      // Auto-submit after selection for immediate feedback
      setTimeout(() => {
        setSubmitted(true);
      }, 500); // Brief delay for visual feedback
    }
  };

  const isCorrect = submitted && selectedOption === correctAnswer;

  return (
    <div className="inline-knowledge-check" role="region" aria-label="Knowledge check">
      <div className="knowledge-check-content">
        <h5>Knowledge Check {position}</h5>
        <p className="knowledge-check-question">{question}</p>

        <ul className="knowledge-check-options">
          {options.map((option, index) => {
            let optionClass = 'knowledge-check-option';
            if (submitted) {
              if (index === correctAnswer) optionClass += ' correct-option';
              if (index === selectedOption && !isCorrect) optionClass += ' incorrect-selected';
            }

            return (
              <li
                key={index}
                className={optionClass}
                onClick={() => !submitted && handleSelect(index)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    !submitted && handleSelect(index);
                  }
                }}
                role="radio"
                aria-checked={index === selectedOption ? 'true' : 'false'}
                tabIndex={0}
              >
                <span className="option-letter" aria-hidden="true">{String.fromCharCode(65 + index)}.</span>
                <span className="option-text">{option}</span>
                {submitted && index === selectedOption && (
                  <span className="option-feedback">
                    {index === correctAnswer ? ' ✓' : ' ✗'}
                  </span>
                )}
              </li>
            );
          })}
        </ul>

        {showExplanation && (
          <div className={`explanation ${isCorrect ? 'correct-explanation' : 'incorrect-explanation'}`}>
            <strong>{isCorrect ? 'Correct!' : 'Not quite.'}</strong> {explanation}
          </div>
        )}

        {submitted && (
          <div className="knowledge-check-next-steps">
            <p>Continue with the module content or <a href={quizLink}>take the full module quiz</a> for comprehensive assessment.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default InlineKnowledgeCheck;
```

### 2. Add CSS Styling for Refined Quiz System
Add the following CSS to your `src/css/custom.css` file to distinguish between inline checks and full quizzes:

```css
/* Authoritative Quiz Styling (Full Module Quiz) */
.authoritative-quiz-section {
  margin-top: 2rem;
  padding: 1.5rem;
  border: 2px solid #28a745;
  border-radius: 8px;
  background-color: #f8fff9;
  box-shadow: 0 4px 8px rgba(40, 167, 69, 0.1);
}

.authoritative-quiz-section h3 {
  color: #2c3e50;
  border-bottom: 2px solid #28a745;
  padding-bottom: 0.5rem;
  margin-top: 0;
  font-size: 1.5rem;
}

/* Inline Knowledge Check Styling */
.inline-knowledge-check {
  margin: 1.5rem 0;
  padding: 1rem;
  border: 1px dashed #ffc107;
  border-radius: 6px;
  background-color: #fff3cd;
  position: relative;
}

.inline-knowledge-check::before {
  content: "Knowledge Check";
  position: absolute;
  top: -0.75rem;
  left: 1rem;
  background-color: #ffc107;
  color: #212529;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: bold;
}

.inline-knowledge-check h5 {
  margin-top: 0;
  color: #856404;
  font-size: 1rem;
}

.knowledge-check-question {
  font-weight: 500;
  margin-bottom: 0.75rem;
}

.knowledge-check-options {
  list-style: none;
  padding: 0;
  margin-bottom: 1rem;
}

.knowledge-check-option {
  padding: 0.5rem;
  margin: 0.25rem 0;
  border: 1px solid #ffeaa7;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  display: flex;
  align-items: center;
}

.knowledge-check-option:hover {
  background-color: #ffeaa7;
}

.knowledge-check-option.correct-option {
  background-color: #d4edda;
  border-color: #28a745;
}

.knowledge-check-option.incorrect-selected {
  background-color: #f8d7da;
  border-color: #dc3545;
}

.knowledge-check-next-steps {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #ffeaa7;
}

.knowledge-check-next-steps a {
  color: #856404;
  font-weight: 500;
  text-decoration: underline;
}

/* Consistent option styling for both components */
.option-letter, .option-letter {
  font-weight: bold;
  margin-right: 0.5rem;
  min-width: 20px;
}

.option-feedback, .option-feedback {
  margin-left: auto;
  font-weight: bold;
  padding-left: 0.5rem;
}

.explanation, .explanation {
  margin-top: 0.5rem;
  padding: 0.5rem;
  border-radius: 4px;
  font-size: 0.9rem;
}

.correct-explanation, .correct-explanation {
  background-color: #d4edda;
  border: 1px solid #c3e6cb;
  color: #155724;
}

.incorrect-explanation, .incorrect-explanation {
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  color: #721c24;
}

.quiz-submit-btn, .quiz-reset-btn {
  padding: 0.5rem 1rem;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 1rem;
}

.quiz-submit-btn:hover, .quiz-reset-btn:hover {
  background-color: #218838;
}
```

### 3. Update Module Content with Inline Knowledge Checks
Add inline knowledge checks to module content at strategic points. Example for a module file:

```markdown
---
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

## Overview

This module introduces you to the fundamental concepts of ROS 2 (Robot Operating System 2), the middleware that serves as the nervous system for robotic applications. You'll learn about nodes, topics, services, and how to create Python agents that communicate with ROS controllers.

## Core Concepts

### Nodes and Communication

In ROS 2, nodes are the fundamental units of computation. Each node is a process that performs computation and communicates with other nodes through messages.

<!-- Inline Knowledge Check -->
import InlineKnowledgeCheck from '@site/src/components/quiz-refined/InlineKnowledgeCheck';

<InlineKnowledgeCheck
  question="What is the fundamental unit of computation in ROS 2?"
  options={[
    "A service",
    "A node",
    "A topic",
    "A parameter"
  ]}
  correctAnswer={1}
  explanation="In ROS 2, nodes are the fundamental units of computation that perform specific tasks and communicate with other nodes."
  quizLink="#module-quiz"
  position={1}
/>

### Topics and Publishers

Topics are named buses over which nodes exchange messages. A publisher node sends messages to a topic, and subscriber nodes receive messages from a topic.

## Next Steps

After completing this module, you'll have the foundational knowledge needed to work with robotic systems and will be ready to explore simulation environments in Module 2.

## Module Quiz

import AuthoritativeQuizComponent from '@site/src/components/quiz-refined/AuthoritativeQuizComponent';

<AuthoritativeQuizComponent
  moduleId="module1"
  title="Module 1 Quiz - ROS 2 Fundamentals"
  quizSourcePath="../quizzes/module1-quiz.md"
/>
```

### 4. Quiz Content Parser (Conceptual)
The actual implementation would require a build-time solution to parse the quiz files from the `quizzes/` directory. This would typically be implemented as:

1. A Docusaurus plugin that processes quiz files during build
2. A custom loader that converts Markdown quiz format to JSON during build
3. Static generation of quiz components from the source files

For the actual implementation, you would need to create a Docusaurus plugin or use a custom Webpack loader to process the quiz files from the `quizzes/` directory during the build process.

## Testing the Setup

1. Start the development server:
```bash
npm run start
```

2. Navigate to a module page with both inline knowledge checks and the end-of-module quiz
3. Verify that:
   - Inline knowledge checks appear within the content flow
   - Inline checks provide immediate feedback
   - The full quiz appears at the end of the module
   - The full quiz uses content from the authoritative source file
   - Visual distinction exists between inline checks and full quiz
   - Links between inline checks and full quiz work correctly

## Deployment
The refined quiz system will be built into the static site during the regular Docusaurus build process:
```bash
npm run build
```

The quiz components will be bundled with the rest of the site and deployed as part of the standard static site deployment process. The build process should include the logic to parse and convert the quiz files from the `quizzes/` directory into the interactive components.