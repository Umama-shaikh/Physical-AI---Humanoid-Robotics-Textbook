import React from 'react';

const QuestionCard = ({ question, selectedOption, onAnswerSelect, submitted, 'aria-describedby': ariaDescribedBy }) => {
  const isCorrect = submitted && selectedOption === question.correctAnswer;
  const isSelected = selectedOption !== undefined;
  const userWasCorrect = isSelected && isCorrect;

  return (
    <div
      className={`question-card ${submitted && isCorrect ? 'correct' : ''} ${submitted && !isCorrect ? 'incorrect' : ''}`}
      role="group"
      aria-labelledby={`question-${question.id}`}
    >
      <h5 id={`question-${question.id}`}>Question {question.id.replace('q', '')}: {question.text}</h5>
      <ul className="question-options" role="radiogroup" aria-describedby={ariaDescribedBy}>
        {question.options.map((option, index) => {
          const isOptionCorrect = index === question.correctAnswer;
          const isOptionSelected = index === selectedOption;
          const optionId = `question-${question.id}-option-${index}`;

          let optionClass = 'option';
          if (submitted) {
            if (isOptionCorrect) optionClass += ' correct-option';
            if (isOptionSelected && !isOptionCorrect) optionClass += ' incorrect-selected';
          }

          return (
            <li
              key={index}
              id={optionId}
              className={optionClass}
              onClick={() => !submitted && onAnswerSelect(index)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  !submitted && onAnswerSelect(index);
                }
              }}
              role="radio"
              aria-checked={isOptionSelected ? 'true' : 'false'}
              tabIndex={0}
              aria-describedby={`question-${question.id}`}
            >
              <span className="option-letter" aria-hidden="true">{String.fromCharCode(65 + index)}.</span>
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
};

export default QuestionCard;