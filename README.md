# Doctor Genie Blog

This repository hosts the source for the **Doctor Genie** tech blog. The site is built with [Jekyll](https://jekyllrb.com/) using the default *minima* theme.

## Running locally

1. Install Ruby and Bundler.
2. Install dependencies:
   ```bash
   bundle install
   ```
3. Serve the site:
   ```bash
   bundle exec jekyll serve
   ```
4. Visit `http://localhost:4000` in your browser.

## Posting articles

Add new markdown files under the `_posts` directory using the naming format `YEAR-MONTH-DAY-title.md`. Each post requires YAML front matter like this:

```markdown
---
layout: post
title: "Your Post Title"
---
```

Commit the file to publish it with the next build.
