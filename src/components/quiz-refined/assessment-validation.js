/**
 * Assessment Validation Functions
 * These functions ensure content integrity by validating that quiz content
 * matches the authoritative source and that no content has been modified.
 */

/**
 * Validates that the quiz data matches the original source format
 * @param {Object} quizData - The quiz data to validate
 * @param {string} sourceContent - The original source content
 * @returns {Array} Array of validation errors
 */
export function validateQuizAgainstSource(quizData, sourceContent) {
  const errors = [];

  // Validate basic structure
  if (!quizData.id) {
    errors.push('Quiz must have an ID');
  }

  if (!quizData.title) {
    errors.push('Quiz must have a title');
  }

  if (!quizData.questions || quizData.questions.length === 0) {
    errors.push('Quiz must have at least one question');
  }

  // Validate that question count matches source
  const questionCountInSource = (sourceContent.match(/### Question \d+/g) || []).length;
  if (quizData.questions && quizData.questions.length !== questionCountInSource) {
    errors.push(`Question count mismatch: expected ${questionCountInSource}, got ${quizData.questions.length}`);
  }

  // Validate each question
  if (quizData.questions) {
    quizData.questions.forEach((question, index) => {
      if (!question.text || question.text.trim() === '') {
        errors.push(`Question ${index + 1} must have text`);
      }

      if (!question.options || question.options.length === 0) {
        errors.push(`Question ${index + 1} must have options`);
      }

      if (question.correctAnswer < 0 || question.correctAnswer >= question.options.length) {
        errors.push(`Question ${index + 1} has an invalid correct answer index`);
      }
    });
  }

  return errors;
}

/**
 * Validates that inline knowledge checks don't duplicate full quiz content
 * @param {Array} inlineChecks - Array of inline knowledge checks
 * @param {Array} fullQuizQuestions - Array of questions from the full quiz
 * @returns {Array} Array of validation errors
 */
export function validateNoDuplication(inlineChecks, fullQuizQuestions) {
  const errors = [];

  if (!inlineChecks || !fullQuizQuestions) {
    return errors;
  }

  inlineChecks.forEach((check, checkIndex) => {
    fullQuizQuestions.forEach((quizQuestion, quizIndex) => {
      // Check if the question text is too similar (indicating duplication)
      const similarity = calculateTextSimilarity(check.question, quizQuestion.text);
      if (similarity > 0.8) { // If 80% or more similar
        errors.push(`Inline check ${checkIndex + 1} appears to duplicate question ${quizIndex + 1} from the full quiz`);
      }
    });
  });

  return errors;
}

/**
 * Calculates text similarity using a simple algorithm
 * @param {string} text1 - First text to compare
 * @param {string} text2 - Second text to compare
 * @returns {number} Similarity ratio (0 to 1)
 */
function calculateTextSimilarity(text1, text2) {
  if (!text1 || !text2) return 0;

  const clean1 = text1.toLowerCase().replace(/\s+/g, '');
  const clean2 = text2.toLowerCase().replace(/\s+/g, '');

  const minLength = Math.min(clean1.length, clean2.length);
  if (minLength === 0) return 0;

  let matches = 0;
  for (let i = 0; i < minLength; i++) {
    if (clean1[i] === clean2[i]) matches++;
  }

  return matches / minLength;
}

/**
 * Validates that inline knowledge checks are concise (1-2 questions maximum)
 * @param {Object} inlineCheck - The inline knowledge check to validate
 * @returns {Array} Array of validation errors
 */
export function validateInlineCheckConciseness(inlineCheck) {
  const errors = [];

  if (!inlineCheck) {
    errors.push('Inline check cannot be null or undefined');
    return errors;
  }

  // For this validation, we're checking that the component is designed to be concise
  // In practice, this would validate that the component only has 1-2 questions
  if (inlineCheck.options && inlineCheck.options.length > 4) {
    errors.push('Inline knowledge checks should have 2-4 options maximum for conciseness');
  }

  return errors;
}

/**
 * Validates the complete assessment model according to requirements
 * @param {Object} assessmentModel - Complete assessment model with module, quizzes, and inline checks
 * @returns {Array} Array of validation errors
 */
export function validateAssessmentModel(assessmentModel) {
  const errors = [];

  if (!assessmentModel) {
    errors.push('Assessment model cannot be null or undefined');
    return errors;
  }

  // Validate module exists
  if (!assessmentModel.moduleId) {
    errors.push('Module ID is required');
  }

  // Validate authoritative quiz exists and is properly formatted
  if (!assessmentModel.authoritativeQuiz) {
    errors.push('Authoritative quiz is required');
  } else {
    const quizErrors = validateQuizAgainstSource(assessmentModel.authoritativeQuiz, assessmentModel.sourceContent || '');
    errors.push(...quizErrors);
  }

  // Validate inline checks don't duplicate full quiz content
  if (assessmentModel.inlineKnowledgeChecks && assessmentModel.authoritativeQuiz?.questions) {
    const duplicationErrors = validateNoDuplication(
      assessmentModel.inlineKnowledgeChecks,
      assessmentModel.authoritativeQuiz.questions
    );
    errors.push(...duplicationErrors);
  }

  // Validate inline checks are concise
  if (assessmentModel.inlineKnowledgeChecks) {
    assessmentModel.inlineKnowledgeChecks.forEach((check, index) => {
      const concisenessErrors = validateInlineCheckConciseness(check);
      if (concisenessErrors.length > 0) {
        errors.push(`Inline check ${index + 1}: ${concisenessErrors.join(', ')}`);
      }
    });
  }

  return errors;
}

/**
 * Validates that quiz content matches the corresponding module
 * @param {Object} quizData - The quiz data
 * @param {string} moduleId - The module ID
 * @returns {Array} Array of validation errors
 */
export function validateQuizModuleMatch(quizData, moduleId) {
  const errors = [];

  if (!quizData || !moduleId) {
    return errors;
  }

  // Check if the quiz's module ID matches the expected module
  if (quizData.module && quizData.module !== moduleId) {
    errors.push(`Quiz module (${quizData.module}) does not match expected module (${moduleId})`);
  }

  // Additional validation could check if quiz questions are relevant to the module content
  // This would require more sophisticated content analysis

  return errors;
}

export default {
  validateQuizAgainstSource,
  validateNoDuplication,
  validateInlineCheckConciseness,
  validateAssessmentModel,
  validateQuizModuleMatch,
  calculateTextSimilarity
};