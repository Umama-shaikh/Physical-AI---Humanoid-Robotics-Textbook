import React, { useState, useEffect } from 'react';

// This component renders the authoritative quiz from pre-processed quiz data
// The quiz data should be pre-processed during build time to avoid client-side file access issues
const AuthoritativeQuizComponent = ({ moduleId, title = "Module Quiz", quizData }) => {
  const [responses, setResponses] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  // Initialize responses state based on quiz questions
  useEffect(() => {
    if (quizData && quizData.questions) {
      const initialResponses = {};
      quizData.questions.forEach((question) => {
        initialResponses[question.id] = null;
      });
      setResponses(initialResponses);
    }
  }, [quizData]);

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
    const resetResponses = {};
    if (quizData && quizData.questions) {
      quizData.questions.forEach((question) => {
        resetResponses[question.id] = null;
      });
    }
    setResponses(resetResponses);
    setSubmitted(false);
  };

  if (loading) {
    return <div className="authoritative-quiz-section">Loading quiz...</div>;
  }

  if (!quizData || !quizData.questions || quizData.questions.length === 0) {
    return <div className="authoritative-quiz-section">Quiz data not available</div>;
  }

  const correctCount = quizData.questions.filter(q =>
    responses[q.id] === q.correctAnswer && submitted
  ).length;

  return (
    <div className="authoritative-quiz-section" role="region" aria-labelledby="quiz-title">
      <h3 id="quiz-title">{title}</h3>
      <div className="quiz-instructions" id="quiz-instructions">
        <p>Complete assessment covering all concepts from this module.</p>
      </div>

      {quizData.questions.map((question, index) => {
        const selectedOption = responses[question.id];
        const isCorrect = submitted && selectedOption === question.correctAnswer;
        const isSelected = selectedOption !== null && selectedOption !== undefined;
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
                <strong>Correct!</strong> {question.explanation || 'Well done!'}
              </div>
            )}

            {submitted && isSelected && !userWasCorrect && (
              <div className="explanation incorrect-explanation" role="status">
                <strong>Incorrect.</strong> {question.explanation || 'Review this concept.'}
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
          <h4>Results: {correctCount} of {quizData.questions.length} correct</h4>
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