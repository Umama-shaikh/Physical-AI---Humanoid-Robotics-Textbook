import { parseQuizMarkdown } from './src/components/quiz-refined/QuizContentParser.js';
import fs from 'fs';

// Read a quiz file to test the format 1 parsing
const quizContent = fs.readFileSync('./quizzes/module1-quiz.md', 'utf8');

console.log('Testing format 1 parsing...');
const result = parseQuizMarkdown(quizContent);

console.log('Parsed result:', JSON.stringify(result, null, 2));
console.log('Number of questions found:', result.questions.length);

if (result.questions.length > 0) {
  console.log('First question:', result.questions[0]);
  console.log('First question correctAnswer:', result.questions[0].correctAnswer);
}