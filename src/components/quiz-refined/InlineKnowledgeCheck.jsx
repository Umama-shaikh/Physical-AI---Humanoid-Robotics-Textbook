import React, { useState } from 'react';

// This component provides a brief inline knowledge check with immediate feedback
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