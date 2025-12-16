/**
 * Build-time Quiz Processor
 * This script would be used during the build process to parse quiz files from the quizzes/ directory
 * and convert them to JSON format that can be imported by the AuthoritativeQuizComponent.
 *
 * In a real implementation, this would be integrated into the Docusaurus build process.
 */

const fs = require('fs');
const path = require('path');
const { parseQuizMarkdown, validateQuizData, formatQuizForComponent } = require('./QuizContentParser');

/**
 * Process all quiz files in the quizzes directory
 * @param {string} quizzesDir - Path to the quizzes directory
 * @param {string} outputDir - Path where processed quiz files will be saved
 */
function processAllQuizzes(quizzesDir, outputDir) {
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Read all files in the quizzes directory
  const files = fs.readdirSync(quizzesDir);

  const quizFiles = files.filter(file =>
    file.endsWith('.md') && file.startsWith('module') && file.includes('-quiz')
  );

  const processedQuizzes = {};

  for (const file of quizFiles) {
    try {
      const filePath = path.join(quizzesDir, file);
      const content = fs.readFileSync(filePath, 'utf8');

      // Parse the quiz markdown
      const quizData = parseQuizMarkdown(content);

      // Validate the parsed data
      const errors = validateQuizData(quizData);
      if (errors.length > 0) {
        console.warn(`Validation errors in ${file}:`, errors);
        continue;
      }

      // Format for component use
      const formattedData = formatQuizForComponent(quizData);

      // Save the processed quiz data
      const outputFileName = file.replace('.md', '.json');
      const outputPath = path.join(outputDir, outputFileName);
      fs.writeFileSync(outputPath, JSON.stringify(formattedData, null, 2));

      processedQuizzes[quizData.id] = formattedData;

      console.log(`Processed quiz: ${file} -> ${outputFileName}`);
    } catch (error) {
      console.error(`Error processing ${file}:`, error.message);
    }
  }

  return processedQuizzes;
}

/**
 * Generate a module that exports all processed quizzes
 * @param {Object} processedQuizzes - Object containing all processed quizzes
 * @param {string} outputDir - Directory to save the generated module
 */
function generateQuizModule(processedQuizzes, outputDir) {
  const moduleContent = `// Auto-generated quiz module
// This file is generated during the build process from quiz markdown files

${Object.entries(processedQuizzes).map(([id, data]) => {
  return `export const ${id.replace(/-/g, '_')}_quiz = ${JSON.stringify(data, null, 2)};`;
}).join('\n\n')}

// Export all quizzes as a single object
export const allQuizzes = {
${Object.keys(processedQuizzes).map(id => {
  return `  "${id}": ${id.replace(/-/g, '_')}_quiz`;
}).join(',\n')}
};
`;

  const outputPath = path.join(outputDir, 'quiz-data.js');
  fs.writeFileSync(outputPath, moduleContent);

  console.log('Generated quiz data module:', outputPath);
}

// Example usage (would be called during build process)
function runBuildProcess() {
  const quizzesDir = path.join(__dirname, '../../../quizzes'); // Adjust path as needed
  const outputDir = path.join(__dirname, '../../../src/data/quizzes-refined'); // Output directory for processed quizzes

  console.log('Starting quiz processing...');
  const processedQuizzes = processAllQuizzes(quizzesDir, outputDir);
  generateQuizModule(processedQuizzes, outputDir);
  console.log('Quiz processing completed.');
}

// Only run if this file is executed directly (not imported)
if (require.main === module) {
  runBuildProcess();
}

module.exports = {
  processAllQuizzes,
  generateQuizModule,
  runBuildProcess
};