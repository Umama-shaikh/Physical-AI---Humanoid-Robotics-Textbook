# Implementation Plan: Module Quiz Page Architecture Correction

**Branch**: `3-quiz-page-architecture` | **Date**: 2025-12-16 | **Spec**: [link](spec.md)
**Input**: Feature specification from `/specs/3-quiz-page-architecture/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Refactor module quiz placement to follow proper textbook structure by creating dedicated quiz pages for each module and removing embedded quizzes from overview pages. The solution will use existing quiz markdown files in the /quizzes directory and position quiz pages as the final item in the sidebar navigation.

## Technical Context

**Language/Version**: JavaScript/TypeScript, Node.js (Docusaurus framework)
**Primary Dependencies**: Docusaurus, React, MDX
**Storage**: N/A (static site generation)
**Testing**: N/A (static content changes)
**Target Platform**: Web (static site)
**Project Type**: Web/single-page application
**Performance Goals**: N/A (static content)
**Constraints**: Must preserve existing quiz content without modification, maintain navigation structure
**Scale/Scope**: 4 modules with quiz content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The implementation follows the core principles by:
- Maintaining separation of concerns (content vs assessment)
- Preserving existing functionality while reorganizing structure
- Using existing technologies (Docusaurus, React components)
- Maintaining backward compatibility with existing content

## Project Structure

### Documentation (this feature)

```text
specs/3-quiz-page-architecture/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
docs/
├── module1/
│   ├── index.md         # Module overview without quizzes
│   └── quiz.md          # New dedicated quiz page
├── module2/
│   ├── index.md         # Module overview without quizzes
│   └── quiz.md          # New dedicated quiz page
├── module3/
│   ├── index.md         # Module overview without quizzes
│   └── quiz.md          # New dedicated quiz page
├── module4/
│   ├── index.md         # Module overview without quizzes
│   └── quiz.md          # New dedicated quiz page
└── ...                  # Other module directories

quizzes/
├── module1-quiz.md      # Source quiz content
├── module2-quiz.md      # Source quiz content
├── module3-quiz.md      # Source quiz content
└── module4-quiz.md      # Source quiz content

src/
└── components/
    └── quiz-refined/    # Quiz components (if needed)

sidebars.js              # Updated sidebar configuration
```

**Structure Decision**: Selected single project structure with Docusaurus documentation organization. Quiz content will be sourced from existing files in /quizzes directory and presented on dedicated pages in each module directory.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |