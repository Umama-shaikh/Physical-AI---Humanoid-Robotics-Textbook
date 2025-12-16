# Data Model: User Entry & Navigation System

## Landing Page Structure

### LandingPage Entity
- **title**: string - "Physical AI & Humanoid Robotics" (required)
- **subtitle**: string - Brief description of the book's purpose (optional)
- **description**: string - Detailed explanation of scope and objectives (required)
- **cta_text**: string - Call-to-action button text (required, default: "Explore Modules")
- **cta_link**: string - URL for the call-to-action (required, default: "/docs")
- **sections**: array of Section objects - Content sections for the landing page
- **created_at**: datetime - Timestamp of creation
- **updated_at**: datetime - Timestamp of last update

### Section Entity
- **id**: string - Unique identifier for the section (required)
- **title**: string - Section title (required)
- **content**: string - Section content in Markdown/MDX format (required)
- **order**: integer - Display order (required)
- **type**: string - Section type (e.g., "hero", "overview", "modules", "getting-started")

## Navigation Configuration

### NavigationConfig Entity
- **logo_url**: string - URL for the logo link (required, default: "/")
- **title_url**: string - URL for the title link (required, default: "/")
- **modules_url**: string - URL for the modules link (required, default: "/docs")
- **show_logo**: boolean - Whether to display the logo (required, default: true)
- **show_title**: boolean - Whether to display the title (required, default: true)
- **show_modules_link**: boolean - Whether to display the modules link (required, default: true)

### NavigationItem Entity
- **id**: string - Unique identifier for the navigation item (required)
- **label**: string - Display text for the navigation item (required)
- **url**: string - Destination URL (required)
- **icon**: string - Icon identifier (optional)
- **order**: integer - Display order in navigation (required)
- **visible**: boolean - Whether the item is visible (required, default: true)

## Route Mapping

### RouteMapping Entity
- **route**: string - The URL route (e.g., "/", "/docs") (required)
- **component**: string - The component or page to render (required)
- **title**: string - Page title for SEO (required)
- **description**: string - Meta description for SEO (optional)
- **requires_auth**: boolean - Whether authentication is required (required, default: false)
- **public**: boolean - Whether the route is publicly accessible (required, default: true)

## Validation Rules

### LandingPage Validation
- Title must be between 10-100 characters
- Description must be between 50-500 characters
- Must have at least one section
- CTA link must be a valid URL or relative path
- Sections must have unique IDs

### NavigationConfig Validation
- Logo URL must be a valid URL or relative path
- Title URL must be a valid URL or relative path
- Modules URL must be a valid URL or relative path
- At least one navigation element must be visible

### NavigationItem Validation
- Label must be between 1-50 characters
- URL must be a valid URL or relative path
- Order must be a non-negative integer
- ID must be unique within the navigation

## State Transitions

### LandingPage States
- **draft**: Content is being created but not published
- **published**: Content is live and accessible
- **archived**: Content is no longer accessible but preserved

### Navigation States
- **enabled**: Navigation item is active and clickable
- **disabled**: Navigation item is visible but not clickable
- **hidden**: Navigation item is not displayed

## Relationships

### LandingPage -> Section
- One-to-many relationship
- LandingPage can have multiple sections
- Each section belongs to exactly one LandingPage

### NavigationConfig -> NavigationItem
- One-to-many relationship
- NavigationConfig can have multiple navigation items
- Each navigation item belongs to exactly one NavigationConfig