# Quickstart Guide: Module Quiz Page Architecture Correction

## Overview
This guide provides step-by-step instructions to refactor module quiz placement from embedded components to dedicated pages.

## Prerequisites
- Node.js and npm installed
- Docusaurus development environment set up
- Access to existing quiz markdown files in /quizzes directory

## Step-by-Step Process

### Step 1: Create Dedicated Quiz Pages
For each module (1-4), create a new quiz.md file:

1. Navigate to each module directory:
   - `docs/module1/`
   - `docs/module2/`
   - `docs/module3/`
   - `docs/module4/`

2. Create a new `quiz.md` file in each directory with the following structure:

```markdown
---
sidebar_position: [position after last chapter]
---

# Module X Quiz

import AuthoritativeQuizComponent from '@site/src/components/quiz-refined/AuthoritativeQuizComponent';
import quizData from '@site/src/data/quizzes-refined/moduleX-quiz.json';

<AuthoritativeQuizComponent
  moduleId={quizData.module}
  title={quizData.title}
  quizData={quizData}
/>
```

### Step 2: Remove Embedded Content from Overview Pages
For each module's index.md file:
1. Remove the "Knowledge Check" section with InlineKnowledgeCheck component
2. Remove the "Module Quiz" section with AuthoritativeQuizComponent
3. Preserve all other educational content (overview, objectives, chapters, etc.)

### Step 3: Update Sidebar Configuration
1. Open `sidebars.js`
2. For each module's sidebar array, add the quiz page as the final item:
```javascript
module1: [
  'module1/index',
  'module1/ros2-fundamentals',
  'module1/python-agents-ros',
  'module1/urdf-humanoids',
  'module1/quiz'  // Add this as the last item
],
```

### Step 4: Verify Implementation
1. Run the development server: `npm run start`
2. Navigate to each module overview page and confirm no quiz content is present
3. Check the sidebar to ensure quiz pages appear as the last item
4. Click on quiz pages to verify they contain the correct content
5. Confirm all quiz content matches the original files in /quizzes directory

## Files to Modify
- `docs/module1/index.md` - Remove embedded content
- `docs/module2/index.md` - Remove embedded content
- `docs/module3/index.md` - Remove embedded content
- `docs/module4/index.md` - Remove embedded content
- `docs/module1/quiz.md` - Create new file
- `docs/module2/quiz.md` - Create new file
- `docs/module3/quiz.md` - Create new file
- `docs/module4/quiz.md` - Create new file
- `sidebars.js` - Update navigation configuration

## Verification Checklist
- [ ] Module overview pages contain no quiz components
- [ ] Each module has a dedicated quiz page
- [ ] Quiz pages appear as the last item in the sidebar
- [ ] Quiz content matches original files exactly
- [ ] All existing chapter links remain functional
- [ ] Navigation flows properly from content to assessment