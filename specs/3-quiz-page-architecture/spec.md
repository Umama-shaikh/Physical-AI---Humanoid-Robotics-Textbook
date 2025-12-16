# Feature Specification: Module Quiz Page Architecture Correction

**Feature Branch**: `3-quiz-page-architecture`
**Created**: 2025-12-16
**Status**: Draft
**Input**: User description: "Title: Module Quiz Page Architecture Correction

Problem Statement:
Module quizzes are currently rendered inside the module overview (index) pages.
This violates the intended book structure and duplicates assessment content.

Intended Architecture:
- Each module must have a dedicated Quiz page
- Quiz pages must appear as the LAST item in the module sidebar
- Quiz content must be sourced from existing markdown files in /quizzes
- Module overview pages must NOT contain quizzes or knowledge checks

Constraints:
- Do not modify quiz question content
- Do not manually edit markdown files
- Do not introduce new quizzes
- Use existing /quizzes/moduleX-quiz.md files
- Preserve all chapters and navigation structure

Success Criteria:
- Module overview pages contain only educational content
- Each module has exactly one Quiz page
- Quiz pages are accessible via sidebar navigation
- Quiz pages appear after all chapters
- No duplicate quizzes exist"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Dedicated Quiz Pages (Priority: P1)

As a learner, I want to access module quizzes from dedicated pages that appear as the last item in the sidebar, so that I can complete assessments after reviewing all educational content without distractions on the overview page.

**Why this priority**: This is the core architectural requirement that addresses the main problem of quiz placement and navigation structure.

**Independent Test**: Can be fully tested by navigating to the module quiz page from the sidebar and verifying it appears as the last item, delivering a clear learning flow from content to assessment.

**Acceptance Scenarios**:

1. **Given** I am viewing a module's sidebar, **When** I look at the navigation order, **Then** the quiz page appears as the last item after all chapter links
2. **Given** I am on a module overview page, **When** I navigate to the quiz page, **Then** I land on a dedicated quiz page with all quiz content

---

### User Story 2 - Clean Module Overview Pages (Priority: P1)

As a learner, I want module overview pages to contain only educational content without embedded quizzes or knowledge checks, so that I can focus on learning objectives and module structure without assessment distractions.

**Why this priority**: This addresses the core problem of duplicate content and improves the learning experience by separating content from assessment.

**Independent Test**: Can be fully tested by visiting module overview pages and verifying they contain only educational content, delivering a cleaner learning experience.

**Acceptance Scenarios**:

1. **Given** I am on a module overview page, **When** I view the page content, **Then** no quiz components or knowledge checks are present
2. **Given** I am reviewing module content, **When** I read the overview page, **Then** I see only learning objectives, chapters, and educational information

---

### User Story 3 - Preserve Existing Quiz Content (Priority: P2)

As a content maintainer, I want the system to source quiz content from existing markdown files in the /quizzes directory, so that I don't lose any existing quiz questions and the content remains unchanged.

**Why this priority**: This ensures content integrity and prevents loss of existing assessment material during the architectural change.

**Independent Test**: Can be fully tested by comparing quiz content before and after the change, delivering content preservation.

**Acceptance Scenarios**:

1. **Given** quiz content exists in /quizzes/moduleX-quiz.md files, **When** I view the new quiz page, **Then** all questions and answers match the original content
2. **Given** I have existing quiz content, **When** I access the new quiz page, **Then** no questions are modified or lost

---

### Edge Cases

- What happens when a module doesn't have an associated quiz file in /quizzes?
- How does the system handle quiz files with formatting errors or missing content?
- What occurs when quiz files are updated - does the quiz page reflect changes automatically?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create a dedicated quiz page for each module that currently has embedded quizzes
- **FR-002**: System MUST position quiz pages as the LAST item in the module sidebar navigation
- **FR-003**: System MUST source quiz content from existing markdown files in the /quizzes directory
- **FR-004**: System MUST remove all quiz components and knowledge checks from module overview pages
- **FR-005**: System MUST preserve all existing quiz question content without modification
- **FR-006**: System MUST maintain all existing chapter links and navigation structure
- **FR-007**: System MUST ensure quiz pages are accessible via the sidebar navigation menu
- **FR-008**: System MUST validate that no duplicate quiz content exists between overview and quiz pages

### Key Entities

- **Module Quiz Page**: A dedicated page that contains all quiz questions for a specific module, sourced from the /quizzes directory
- **Module Overview Page**: The main page for each module that contains educational content but no assessment components
- **Sidebar Navigation**: The navigation menu that displays chapters and now includes the quiz page as the final item

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Module overview pages contain 0 quiz components or knowledge checks after implementation
- **SC-002**: Each module has exactly 1 dedicated quiz page accessible from the sidebar
- **SC-003**: Quiz pages appear as the last item in the sidebar navigation for each module
- **SC-004**: 100% of quiz content from /quizzes directory is preserved without modification
- **SC-005**: All existing chapter links and navigation structure remain unchanged
- **SC-006**: Users can navigate from module overview to dedicated quiz page without content duplication