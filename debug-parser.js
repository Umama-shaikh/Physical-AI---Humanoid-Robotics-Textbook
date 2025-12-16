const fs = require('fs');
const path = require('path');

// Read a quiz file to debug the format
const quizContent = fs.readFileSync(path.join(__dirname, 'quizzes', 'module1-quiz.md'), 'utf8');

console.log('Quiz content:');
console.log(quizContent.substring(0, 1000)); // First 1000 characters

// Try to match the questions
const questionRegex = /### Question (\d+)\s*\n([^\n\r#]+(?:\n(?!\n### Question|\n##|\n[A-Z]\)).)*)/g;
let match;
const matches = [];

while ((match = questionRegex.exec(quizContent)) !== null) {
  console.log('Found match:', match[0].substring(0, 200)); // First 200 chars of match
  matches.push(match);
}

console.log('Number of matches found:', matches.length);

// Try a simpler approach
const simpleRegex = /### Question (\d+)[\s\S]*?(?=\n### Question \d+|\n##|$)/g;
let simpleMatch;
const simpleMatches = [];

while ((simpleMatch = simpleRegex.exec(quizContent)) !== null) {
  console.log('Simple match:', simpleMatch[0].substring(0, 300)); // First 300 chars
  simpleMatches.push(simpleMatch);
}

console.log('Simple matches found:', simpleMatches.length);