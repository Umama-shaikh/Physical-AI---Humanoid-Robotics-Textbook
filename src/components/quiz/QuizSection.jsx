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
    <div className="quiz-section" role="region" aria-labelledby="quiz-title">
      <h3 id="quiz-title">{title}</h3>
      <div className="quiz-instructions" id="quiz-instructions">
        <p>Test your understanding of the key concepts from this module.</p>
      </div>

      {questions.map((question, index) => (
        <QuestionCard
          key={question.id}
          question={question}
          selectedOption={responses[question.id]}
          onAnswerSelect={(optionIndex) => handleAnswerSelect(question.id, optionIndex)}
          submitted={submitted}
          aria-describedby="quiz-instructions"
        />
      ))}

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

export default QuizSection;