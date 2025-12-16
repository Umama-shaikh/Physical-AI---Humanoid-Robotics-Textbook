/**
 * QuizContentParser - Utility to convert Markdown quiz format to structured data
 * This utility is designed to be used during the build process to parse quiz files
 * from the quizzes/ directory and convert them to structured data for use in components.
 */

/**
 * Parses a quiz Markdown content into structured data
 * @param {string} markdownContent - The raw Markdown content of the quiz
 * @returns {Object} Structured quiz data with questions and metadata
 */
export const parseQuizMarkdown = (markdownContent) => {
  // Extract frontmatter if present
  const frontmatterMatch = markdownContent.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)/);
  let frontmatter = {};
  let content = markdownContent;

  if (frontmatterMatch) {
    const frontmatterStr = frontmatterMatch[1];
    content = frontmatterMatch[2] || '';

    // Parse frontmatter properties
    const lines = frontmatterStr.split('\n');
    lines.forEach(line => {
      const [key, ...valueParts] = line.split(':');
      if (key && valueParts.length > 0) {
        const cleanKey = key.trim();
        const cleanValue = valueParts.join(':').trim().replace(/^["']|["']$/g, '');
        frontmatter[cleanKey] = cleanValue;
      }
    });
  }

  // Extract title from the main heading if not in frontmatter
  const titleMatch = content.match(/^# (.+)$/m);
  const quizTitle = frontmatter.title || (titleMatch ? titleMatch[1] : 'Module Quiz');

  // Try to parse with the first format (with ** and - options)
  let questions = parseFormat1(content);

  // If no questions found with first format, try the second format (without ** and without -)
  if (questions.length === 0) {
    questions = parseFormat2(content);
  }

  return {
    id: frontmatter.id || extractModuleId(quizTitle),
    title: quizTitle,
    module: frontmatter.module || extractModuleId(quizTitle),
    questions: questions,
    instructions: extractInstructions(content)
  };
};

/**
 * Parse format 1: ### Question N\n**Question text**\n- A) Option\n- B) Option\n- C) Option\n- D) Option\n\nAnswer: [Letter]
 */
const parseFormat1 = (content) => {
  // Find all questions with their full content
  const questionRegex = /### Question (\d+)[\s\S]*?\*\*([^\*]+)\*\*[\s\S]*?(?=\n\nAnswer:|\n### Question|\n##|$)/g;
  const questions = [];
  let match;

  while ((match = questionRegex.exec(content)) !== null) {
    const questionNumber = match[1];
    const fullQuestionText = match[0];
    const questionText = match[2] ? match[2].trim() : '';

    // Extract options in format "- A) option"
    const optionRegex = /- ([A-D]\))\s*([^\n\r]+)/g;
    const options = [];
    let optionMatch;
    const optionLetterToIndex = {};
    let optionIndex = 0;

    // Create a new RegExp object for each question to avoid state issues
    const tempOptionRegex = /- ([A-D]\))\s*([^\n\r]+)/g;
    while ((optionMatch = tempOptionRegex.exec(fullQuestionText)) !== null) {
      options.push(optionMatch[2].trim());
      const letter = optionMatch[1].charAt(0); // Get the letter A, B, C, or D
      optionLetterToIndex[letter] = optionIndex;
      optionIndex++;
    }

    // Extract answer from the "Answer: [Letter]" part that follows
    const answerPattern = new RegExp(`### Question ${questionNumber}[\\s\\S]*?Answer:\\s*([A-D])`, 'i');
    const answerMatch = content.match(answerPattern);
    let correctAnswerIndex = -1;

    if (answerMatch && answerMatch[1]) {
      const answerLetter = answerMatch[1];
      if (optionLetterToIndex.hasOwnProperty(answerLetter)) {
        correctAnswerIndex = optionLetterToIndex[answerLetter];
      }
    }

    if (options.length > 0) {
      questions.push({
        id: `q${questionNumber}`,
        text: questionText,
        options: options,
        correctAnswer: correctAnswerIndex
      });
    }
  }

  return questions;
};

/**
 * Parse format 2: ### Question N\nQuestion text\nA) Option\nB) Option\nC) Option\nD) Option\n\n[Answers section at end]
 */
const parseFormat2 = (content) => {
  // Extract the main questions section and the answers section
  const answersSection = content.match(/## Answers[\s\S]*$/);
  const mainContent = answersSection ? content.replace(/## Answers[\s\S]*$/, '') : content;

  // Find all questions in the format "### Question N\nQuestion text\nA) Option\nB) Option..."
  const questionRegex = /### Question (\d+)\s*\n([^\n\r]+(?:\n(?!\n### Question|\n##)[^\n\r#]*)*)/g;
  const questions = [];
  let match;

  while ((match = questionRegex.exec(mainContent)) !== null) {
    const questionNumber = match[1];
    const fullQuestionText = match[0];
    const questionText = match[2] ? match[2].split('\n')[0].trim() : '';

    // Extract options in format "A) option", "B) option", etc.
    const optionRegex = /([A-Z])\)\s*([^\n\r]+)/g;
    const options = [];
    let optionMatch;
    const optionLetterToIndex = {};
    let optionIndex = 0;

    while ((optionMatch = optionRegex.exec(fullQuestionText)) !== null) {
      options.push(optionMatch[2].trim());
      optionLetterToIndex[optionMatch[1]] = optionIndex;
      optionIndex++;
    }

    let correctAnswerIndex = -1;

    // Extract answer from answers section
    if (answersSection && answersSection[0]) {
      // Look for "N. [Letter])" or "N. [Letter] [option text]"
      const answerPattern = new RegExp(`${questionNumber}\\.\\s*([A-Z])[\\)\\s]([^\n\r]*)`, 'i');
      const answerMatch = answersSection[0].match(answerPattern);

      if (answerMatch) {
        const answerLetter = answerMatch[1];
        if (optionLetterToIndex.hasOwnProperty(answerLetter)) {
          correctAnswerIndex = optionLetterToIndex[answerLetter];
        } else {
          // Try to match by option text
          const answerText = answerMatch[2].trim();
          for (let i = 0; i < options.length; i++) {
            if (options[i].includes(answerText) || answerText.includes(options[i])) {
              correctAnswerIndex = i;
              break;
            }
          }
        }
      }
    }

    if (options.length > 0) {
      questions.push({
        id: `q${questionNumber}`,
        text: questionText,
        options: options,
        correctAnswer: correctAnswerIndex
      });
    }
  }

  return questions;
};

/**
 * Extracts module ID from quiz title
 * @param {string} title - The quiz title
 * @returns {string} Module ID
 */
const extractModuleId = (title) => {
  const moduleMatch = title.match(/Module (\d+)/i);
  return moduleMatch ? `module${moduleMatch[1]}` : 'unknown';
};

/**
 * Extracts general instructions from the quiz content
 * @param {string} content - The quiz content without frontmatter
 * @returns {string} Instructions section content
 */
const extractInstructions = (content) => {
  const instructionsMatch = content.match(/## Instructions\s*\n([\s\S]*?)(?=\n## Questions|\n### Question)/);
  if (instructionsMatch) {
    return instructionsMatch[1].trim();
  }
  return '';
};

/**
 * Validates the parsed quiz data against requirements
 * @param {Object} quizData - The parsed quiz data
 * @returns {Array} Array of validation errors
 */
export const validateQuizData = (quizData) => {
  const errors = [];

  if (!quizData.id) {
    errors.push('Quiz must have an ID');
  }

  if (!quizData.title) {
    errors.push('Quiz must have a title');
  }

  if (!quizData.questions || quizData.questions.length === 0) {
    errors.push('Quiz must have at least one question');
  } else {
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
};

/**
 * Converts the parsed quiz data to a format suitable for the AuthoritativeQuizComponent
 * @param {Object} quizData - The parsed quiz data
 * @returns {Object} Formatted data for the component
 */
export const formatQuizForComponent = (quizData) => {
  return {
    id: quizData.id,
    title: quizData.title,
    questions: quizData.questions.map((q, index) => ({
      id: q.id || `q${index + 1}`,
      text: q.text,
      options: q.options,
      correctAnswer: q.correctAnswer,
      explanation: q.explanation || `Explanation for question ${index + 1}`
    }))
  };
};

export default {
  parseQuizMarkdown,
  validateQuizData,
  formatQuizForComponent
};