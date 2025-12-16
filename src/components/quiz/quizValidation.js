/**
 * Quiz data validation functions for the Physical AI & Humanoid Robotics textbook
 */

/**
 * Validates a quiz data structure
 * @param {Object} quizData - The quiz data to validate
 * @returns {Array} Array of validation errors, empty if valid
 */
export const validateQuizData = (quizData) => {
  const errors = [];

  if (!quizData) {
    errors.push('Quiz data is required');
    return errors;
  }

  // Validate moduleId
  if (!quizData.moduleId || typeof quizData.moduleId !== 'string') {
    errors.push('moduleId is required and must be a string');
  }

  // Validate title
  if (!quizData.title || typeof quizData.title !== 'string') {
    errors.push('title is required and must be a string');
  }

  // Validate questions array
  if (!quizData.questions || !Array.isArray(quizData.questions)) {
    errors.push('questions is required and must be an array');
  } else {
    if (quizData.questions.length < 1 || quizData.questions.length > 10) {
      errors.push('questions array must contain between 1 and 10 questions');
    }

    // Validate each question
    quizData.questions.forEach((question, index) => {
      const questionErrors = validateQuestion(question, index);
      errors.push(...questionErrors);
    });
  }

  // Validate instructions (optional)
  if (quizData.instructions && typeof quizData.instructions !== 'string') {
    errors.push('instructions must be a string if provided');
  }

  return errors;
};

/**
 * Validates a single question object
 * @param {Object} question - The question to validate
 * @param {number} index - The index of the question for error context
 * @returns {Array} Array of validation errors, empty if valid
 */
export const validateQuestion = (question, index) => {
  const errors = [];
  const prefix = `Question ${index + 1}: `;

  if (!question) {
    errors.push(`${prefix}Question is required`);
    return errors;
  }

  // Validate id
  if (!question.id || typeof question.id !== 'string') {
    errors.push(`${prefix}id is required and must be a string`);
  }

  // Validate type
  if (!question.type || typeof question.type !== 'string') {
    errors.push(`${prefix}type is required and must be a string`);
  } else if (!['multiple-choice', 'true-false'].includes(question.type)) {
    errors.push(`${prefix}type must be either 'multiple-choice' or 'true-false'`);
  }

  // Validate text
  if (!question.text || typeof question.text !== 'string' || question.text.trim() === '') {
    errors.push(`${prefix}text is required and must be a non-empty string`);
  }

  // Validate options (for multiple-choice)
  if (question.type === 'multiple-choice') {
    if (!question.options || !Array.isArray(question.options)) {
      errors.push(`${prefix}options is required for multiple-choice questions and must be an array`);
    } else {
      if (question.options.length < 2 || question.options.length > 6) {
        errors.push(`${prefix}options array must contain between 2 and 6 options`);
      }

      question.options.forEach((option, optIndex) => {
        if (typeof option !== 'string' || option.trim() === '') {
          errors.push(`${prefix}option ${optIndex + 1} must be a non-empty string`);
        }
      });
    }
  }

  // Validate correctAnswer
  if (question.correctAnswer === undefined || question.correctAnswer === null) {
    errors.push(`${prefix}correctAnswer is required`);
  } else {
    if (question.type === 'multiple-choice') {
      if (!Number.isInteger(question.correctAnswer) ||
          question.correctAnswer < 0 ||
          question.correctAnswer >= (question.options ? question.options.length : 0)) {
        errors.push(`${prefix}correctAnswer must be a valid index for the options array`);
      }
    } else if (question.type === 'true-false') {
      if (typeof question.correctAnswer !== 'boolean') {
        errors.push(`${prefix}correctAnswer must be a boolean for true-false questions`);
      }
    }
  }

  // Validate explanation
  if (!question.explanation || typeof question.explanation !== 'string' || question.explanation.trim() === '') {
    errors.push(`${prefix}explanation is required and must be a non-empty string`);
  }

  return errors;
};

/**
 * Validates a user response
 * @param {Object} response - The user response to validate
 * @param {Array} questionOptions - The options for the question being answered
 * @returns {Array} Array of validation errors, empty if valid
 */
export const validateUserResponse = (response, questionOptions) => {
  const errors = [];

  if (!response) {
    errors.push('Response is required');
    return errors;
  }

  // Validate questionId
  if (!response.questionId || typeof response.questionId !== 'string') {
    errors.push('questionId is required and must be a string');
  }

  // Validate selectedOption
  if (response.selectedOption === undefined || response.selectedOption === null) {
    errors.push('selectedOption is required');
  } else {
    // For multiple-choice, check if it's a valid index
    if (Array.isArray(questionOptions)) {
      if (!Number.isInteger(response.selectedOption) ||
          response.selectedOption < 0 ||
          response.selectedOption >= questionOptions.length) {
        errors.push('selectedOption must be a valid index for the question options');
      }
    }
  }

  return errors;
};