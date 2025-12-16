/**
 * Assessment Navigation Utilities
 * Functions to manage navigation between inline knowledge checks and full quizzes
 */

/**
 * Generates the correct link to the full module quiz
 * @param {string} moduleId - The ID of the module
 * @returns {string} The link to the full module quiz
 */
export function generateFullQuizLink(moduleId) {
  // Return an anchor link that points to the full quiz section at the end of the module
  // This assumes the full quiz section has an ID of "module-quiz" or similar
  return `#module-quiz`;
}

/**
 * Generates a link to a specific question within the full quiz
 * @param {string} moduleId - The ID of the module
 * @param {number} questionIndex - The index of the question
 * @returns {string} The link to the specific question
 */
export function generateQuestionLink(moduleId, questionIndex) {
  return `#question-${questionIndex + 1}`;
}

/**
 * Finds the position of inline knowledge checks within a module's content
 * @param {Array} contentBlocks - Array of content blocks in the module
 * @returns {Array} Array of positions where inline knowledge checks occur
 */
export function findInlineCheckPositions(contentBlocks) {
  if (!contentBlocks) return [];

  const positions = [];
  contentBlocks.forEach((block, index) => {
    if (block.type === 'inline-knowledge-check' ||
        block.component === 'InlineKnowledgeCheck' ||
        (block.props && block.props.question)) {
      positions.push(index);
    }
  });

  return positions;
}

/**
 * Validates that navigation links are properly configured
 * @param {Object} inlineCheck - The inline knowledge check component
 * @param {string} moduleId - The module ID
 * @returns {Array} Array of validation errors
 */
export function validateNavigationLinks(inlineCheck, moduleId) {
  const errors = [];

  if (!inlineCheck) {
    errors.push('Inline check cannot be null or undefined');
    return errors;
  }

  // Check if the quiz link is properly formatted
  if (!inlineCheck.quizLink) {
    errors.push('Inline check must have a quizLink property');
  } else if (!inlineCheck.quizLink.includes('#module-quiz') && !inlineCheck.quizLink.includes(moduleId)) {
    errors.push('Quiz link should reference the full module quiz');
  }

  return errors;
}

/**
 * Updates navigation state when moving between assessment types
 * @param {string} fromType - The type of assessment being navigated from ('inline' or 'full')
 * @param {string} toType - The type of assessment being navigated to ('inline' or 'full')
 * @param {Object} additionalData - Additional data for the navigation
 * @returns {Object} Navigation state update object
 */
export function updateNavigationState(fromType, toType, additionalData = {}) {
  const navigationState = {
    from: fromType,
    to: toType,
    timestamp: new Date().toISOString(),
    ...additionalData
  };

  // Log navigation for analytics purposes
  console.log(`Navigation: ${fromType} → ${toType}`, additionalData);

  return navigationState;
}

/**
 * Gets the appropriate quiz link based on context
 * @param {string} moduleId - The module ID
 * @param {string} context - The context where the link is needed ('inline-check', 'module-end', 'table-of-contents', etc.)
 * @returns {string} The appropriate quiz link for the context
 */
export function getContextualQuizLink(moduleId, context = 'inline-check') {
  switch (context) {
    case 'inline-check':
      // From inline check, link to full quiz at module end
      return '#module-quiz';
    case 'module-end':
      // From module end, link to first question of full quiz
      return '#question-1';
    case 'table-of-contents':
      // From table of contents, link to quiz section
      return '#module-quiz';
    case 'previous-section':
      // From previous section link, go to quiz
      return '#module-quiz';
    default:
      return '#module-quiz';
  }
}

/**
 * Validates the relationship between module content and its quiz
 * @param {Object} moduleContent - The module content structure
 * @param {Object} quizData - The quiz data
 * @returns {Array} Array of validation errors
 */
export function validateModuleQuizRelationship(moduleContent, quizData) {
  const errors = [];

  if (!moduleContent) {
    errors.push('Module content is required');
  }

  if (!quizData) {
    errors.push('Quiz data is required');
  }

  // Check if the module ID matches the quiz's expected module
  if (moduleContent.id && quizData.module && moduleContent.id !== quizData.module) {
    errors.push(`Module ID (${moduleContent.id}) does not match quiz module (${quizData.module})`);
  }

  // Check if the quiz has questions
  if (quizData.questions && quizData.questions.length === 0) {
    errors.push('Quiz must have at least one question');
  }

  return errors;
}

export default {
  generateFullQuizLink,
  generateQuestionLink,
  findInlineCheckPositions,
  validateNavigationLinks,
  updateNavigationState,
  getContextualQuizLink,
  validateModuleQuizRelationship
};