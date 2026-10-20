# Feature briefs

One markdown file per shipped, user-facing feature or new model-family support. A separate video-maker agent reads a brief and produces a promo video from it. This directory holds the briefs only — not video files, shot lists, or edit notes.

`CHANGELOG.md` stays the chronological product log (what landed, in order). A feature brief is the deeper intro: who it is for, what a person can actually do, how to try it, and what is out of scope.

Write only behavior that shipped. Verify names, sizes, and defaults against the merge. Do not invent metrics, UI, or modalities.

## Filename / 文件名

`YYYY-MM-DD-short-kebab-slug.md`

The date is the merge or ship day when that day is known.

## Required sections / 必备小节

Copy this outline into each new brief. Keep the English headings so another agent can find them. Body text may be Chinese, English, or both (Chinese body plus an English one-line pitch is the usual split when the video will be narrated in Chinese).

```markdown
## Title

## One-line pitch

## Who it is for / when to care

## What shipped (user-visible behavior)

## How to try it

## Limits / out of scope

## Suggested video beats

## Links
```

- **Title** — product name a viewer would recognize.
- **One-line pitch** — one sentence. Say what is new and where it runs.
- **Who it is for / when to care** — the user and the situation, not an architecture tour.
- **What shipped (user-visible behavior)** — what changes after this merge: load path, defaults, catalog rows, errors a person can see.
- **How to try it** — concrete download, load, and demo steps (commands or in-app clicks that exist today).
- **Limits / out of scope** — what this ship does not do, including paths the video must not demo.
- **Suggested video beats** — bullet talking points only. No shot list, no timing, no editing instructions.
- **Links** — pull request, model repos, and related docs (including this file’s sibling briefs when they matter).
