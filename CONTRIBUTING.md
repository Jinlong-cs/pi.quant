# Contributing to pi.quant

Read `AGENTS.md` before editing. The `.agents/` directory is the canonical
source for repository workflows shared by Codex, Claude Code, and other agents.

Public changes must preserve the model/calibration/evidence boundary. Do not
commit datasets, checkpoints, engines, raw logs, credentials, generated videos,
or local environments.

Only stable tests that protect public behavior belong in `tests/`. Temporary
probes, one-off hardware scripts, and experiment matrices stay in an external
task artifact directory and are never added to a feature PR.

Every release increment is an independent feature PR. Do not merge or tag from
an agent session unless the task explicitly grants that authority.
