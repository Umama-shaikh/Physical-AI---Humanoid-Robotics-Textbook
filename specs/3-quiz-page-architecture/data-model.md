# Data Model: Module Quiz Page Architecture Correction

## Entities

### Module Quiz Page
- **Purpose**: Dedicated page containing all quiz questions for a specific module
- **Content**: Quiz questions sourced from /quizzes directory
- **Relationships**: Linked to one specific module in the sidebar navigation
- **Validation**: Must contain all questions from the corresponding quiz file without modification

### Module Overview Page
- **Purpose**: Main page for each module containing educational content only
- **Content**: Overview, learning objectives, chapters, prerequisites, next steps (no assessment content)
- **Relationships**: Links to chapters and quiz page in sidebar navigation
- **Validation**: Must not contain any quiz components or knowledge checks

### Sidebar Navigation Item
- **Purpose**: Navigation menu item for the quiz page
- **Fields**:
  - label: "Quiz" or "Module X Quiz"
  - to: path to quiz page (e.g., "/docs/module1/quiz")
  - position: must be last in module's sidebar section
- **Relationships**: Part of module's sidebar section, follows all chapter items

## Data Flow

### Quiz Content Flow
1. Source: /quizzes/moduleX-quiz.md (original quiz markdown files)
2. Process: Build-time processing converts to JSON format in /src/data/quizzes-refined/
3. Consume: Quiz page imports and renders the processed quiz data

### Navigation Flow
1. Define: Quiz page created in each module directory
2. Register: Sidebar configuration updated to include quiz page as final item
3. Render: Docusaurus generates navigation with proper ordering

## State Transitions

### Module Content State
- **Before**: Module overview contains embedded quizzes and knowledge checks
- **During**: Embedded content is removed, dedicated quiz page is created
- **After**: Overview contains only educational content, quiz page contains all assessment content

## Validation Rules

### Content Validation
- Each module overview page must have 0 quiz components
- Each quiz page must source content from the corresponding file in /quizzes directory
- All quiz content must match the original files exactly (no modifications)

### Navigation Validation
- Quiz page must appear as the last item in each module's sidebar section
- All existing chapter links must remain unchanged
- Sidebar hierarchy must maintain proper ordering (chapters → quiz)

## Constraints

### Content Constraints
- Quiz questions must not be modified during migration
- All existing educational content must be preserved on overview pages
- No duplicate content between overview and quiz pages

### Navigation Constraints
- Quiz pages must be accessible via sidebar navigation
- Sidebar ordering must follow: overview → chapters → quiz
- Existing navigation structure for chapters must remain unchanged