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

v0.3 adds the narrow temporal/WAM boundary. FastWAM is an explicit source
adapter, not a universal model registry: semantic inventory, temporal axes,
and execution callbacks are injected by the caller. Teacher-forced execution
and world-latent capture are required only when a study claims those signals;
missing source capabilities remain pending and cannot be replaced by synthetic
or fabricated outputs.

v0.4 adds the target compiler evidence boundary. TensorRT/ONNX work belongs in
optional integrations and external Task Contract artifacts. Capability probes,
build records, layer inspection, stage timing, server/client timing and
closed-loop promotion are separate evidence lanes. Do not commit hardware
scripts, generated engines, ONNX assets, traces, logs or machine-specific
identity files.

v0.5 adds deterministic mixed-precision search and pending-first promotion.
Search plans require four disjoint data splits, measured source recovery and
target-local cost, explicit budgets, and matched FP/broad/manual controls.
Source and target Pareto fronts are distinct. Gate40/full400 require external
approval, and no agent or machine record can assign human acceptance. Keep
plans, candidates, experiment runners, manifests, and results outside Git.
