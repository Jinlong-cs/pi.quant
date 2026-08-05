# `.agents/` — agent-agnostic source of truth

This directory is the canonical location for pi.quant workflows shared by
Codex, Claude Code, Cursor, and other coding agents.

```text
.agents/
├── README.md
└── skills/
    └── <skill-name>/SKILL.md
```

Always edit the canonical files under `.agents/`. `.claude/skills` is a
committed relative symlink for Claude discovery and must not become a second
copy. New skills need YAML frontmatter, explicit scope, prerequisites, steps,
validation gates, evidence outputs, and promotion boundaries.

The skills describe how to operate the library; they do not replace the
versioned Python contracts or decide human acceptance.
