# Quickstart Guide: User Entry & Navigation System

## Overview
This guide provides a quick setup for the landing page and navigation system for the Physical AI & Humanoid Robotics book.

## Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager
- Docusaurus CLI installed globally (`npm install -g @docusaurus/cli`)

## Setup Steps

### 1. Create the Landing Page
Create a new file at `src/pages/index.md` with the following content:

```markdown
---
title: Physical AI & Humanoid Robotics
---

# Physical AI & Humanoid Robotics

## A Beginner-Friendly Textbook for Teaching Physical AI & Humanoid Robotics

This comprehensive textbook provides a practical approach to learning robotics, artificial intelligence, and humanoid systems. Whether you're a student, researcher, or enthusiast, this book guides you through the essential concepts and implementation techniques needed to build autonomous humanoid robots.

### What You'll Learn

- ROS 2 fundamentals and advanced concepts
- Simulation environments for robotics development
- AI perception with Isaac ROS and VSLAM
- Navigation systems for humanoid robots
- Voice-to-action processing with Whisper
- LLM cognitive planning for robots
- Complete autonomous system integration

### Getting Started

[Explore Modules](./docs/intro)
```

### 2. Configure Navigation in docusaurus.config.js

Update your `docusaurus.config.js` file to ensure proper navigation behavior:

```javascript
// docusaurus.config.js
module.exports = {
  // ... other config
  navbar: {
    title: 'Physical AI & Humanoid Robotics',
    logo: {
      alt: 'Physical AI Book Logo',
      src: 'img/logo.svg',
    },
    items: [
      {
        type: 'docSidebar',
        sidebarId: 'tutorialSidebar',
        position: 'left',
        label: 'Modules',
      },
      {
        href: 'https://github.com/your-organization/physical-ai-book',
        label: 'GitHub',
        position: 'right',
      },
    ],
  },
  // ... other config
};
```

### 3. Update Sidebar Configuration

Ensure your `sidebars.js` includes an intro page that the landing page can link to:

```javascript
// sidebars.js
module.exports = {
  tutorialSidebar: [
    'intro',
    // ... other sidebar items
  ],
};
```

### 4. Create Intro Page

Create `docs/intro.md` as the entry point for the documentation:

```markdown
---
sidebar_position: 1
---

# Welcome to the Modules

Welcome to the Physical AI & Humanoid Robotics textbook modules. This section contains all the detailed content covering ROS 2 fundamentals, simulation environments, AI perception, navigation, and more.

## Getting Started

Choose a module from the sidebar to begin your learning journey:

- **Module 1**: ROS 2 Fundamentals
- **Module 2**: Simulation Environments
- **Module 3**: AI Perception
- **Module 4**: Vision-Language-Action Systems

Each module contains theoretical concepts, practical examples, and hands-on exercises to reinforce your learning.
```

## Testing the Setup

1. Start the development server:
```bash
npm run start
```

2. Verify navigation elements:
   - Visit `http://localhost:3000` - should show the landing page
   - Click the site logo - should return to the landing page
   - Click the site title - should return to the landing page
   - Click "Modules" in the navbar - should go to the documentation

3. Check for 404 errors by clicking all navigation elements

## Deployment

For GitHub Pages deployment:

1. Set the correct `baseUrl` and `organizationName`/`projectName` in `docusaurus.config.js`
2. Build the site: `npm run build`
3. Deploy to GitHub Pages following the Docusaurus deployment guide

## Troubleshooting

### Navigation not working
- Ensure `docusaurus.config.js` has correct navbar configuration
- Verify that sidebar items exist and are properly named

### 404 errors
- Check that all navigation links point to existing pages
- Verify file paths and extensions are correct

### Logo or title not linking to homepage
- Confirm that the root route (`/`) is properly configured
- Check that the homepage file exists at `src/pages/index.js` or `src/pages/index.md`