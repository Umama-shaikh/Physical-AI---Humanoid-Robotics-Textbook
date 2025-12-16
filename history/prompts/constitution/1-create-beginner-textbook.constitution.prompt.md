---
id: 1
title: Create Beginner Textbook
stage: constitution
date: 2025-12-12
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user:
command: /sp.constitution
labels: [constitution,physical-ai,humanoid-robotics,textbook]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
Project: Beginner-Friendly Textbook for Teaching Physical AI & Humanoid Robotics
Platform Requirement: Must be created and structured using Docusaurus
Process Requirement: Must follow Spec-Driven Creation using Spec Kit

Core principles:
- Beginner Accessibility: All concepts explained clearly with minimal prerequisites.
- Technical Accuracy: Robotics, AI, and physics content must be validated against reliable primary sources.
- Pedagogical Clarity: Concepts introduced gradually with examples and illustrations.
- Modularity: Each chapter structured as standalone markdown within Docusaurus.
- Reproducibility: All examples, diagrams, and instructions must be reproducible.
- Open-Source Quality: Consistent formatting, clean structure, and contributor-friendly documentation.

Key standards:
- Citation style: APA or IEEE (choose one and apply consistently).
- Source types: Robotics/AI textbooks, peer-reviewed papers, technical documentation.
- Beginner-friendly writing: Flesch–Kincaid grade 8–10; all jargon defined on first use.
- Docusaurus structure:
  - Chapters stored in /docs/
  - Sidebar navigation required
  - Each chapter in its own markdown file
  - Versioning optional (off by default)
- Visual standards:
  - Diagrams must be simple, clear, and labeled
  - Images must be original or open-license only

Constraints:
- Book scope: Introductory textbook on Physical AI & Humanoid Robotics.
- Format: Entire book built within a Docusaurus project.
- Length: 8–12 chapters (~20–40 pages equivalent).
- Content level: No heavy math unless essential; focus on intuition and basics.
- Technical boundaries: Covers fundamentals; excludes advanced robotics algorithms unless simplified.

Success criteria:
- The Docusaurus site builds successfully without errors.
- All chapters follow a clear, consistent, beginner-friendly structure.
- All technical content is verified, accurate, and sourced.
- Includes diagrams, examples, and simple explanations.
- Zero plagiarism; originality maintained.
- Content passes clarity and accuracy checks.
- The final book forms a complete beginner-friendly introduction to the topic.
```

## Response snapshot

Constitution created for the Beginner-Friendly Textbook project on Physical AI & Humanoid Robotics. The constitution includes six core principles (Beginner Accessibility, Technical Accuracy, Pedagogical Clarity, Modularity and Standalone Structure, Reproducibility, and Open-Source Quality Standards), content standards for citation and accessibility, development workflow guidelines, and governance rules. The document is structured for use with Docusaurus and follows all specified requirements including APA citation style, Flesch-Kincaid grade 8-10 writing level, and visual standards for diagrams and images.

## Outcome

- ✅ Impact: Created initial project constitution with all required principles and standards
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive constitution
- 🔁 Next prompts: Proceed with creating feature specifications using /sp.specify
- 🧠 Reflection: Constitution provides strong foundation for textbook development with clear accessibility and technical standards

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - all required elements included
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating the first chapter specification