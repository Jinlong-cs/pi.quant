# Repository First Principles

## Authority

* Read this file before editing, staging, committing, or opening/updating a pull request in this repository.
* Treat these repository boundaries as the first review gate. A working experiment is not automatically suitable repository
  content.
* Keep the public repository focused on reusable quantization functionality, stable contracts, and concise user-facing
  documentation.

## Commit And Pull Request Boundary

* Commit only reusable product code, versioned contracts, public recipes, concise documentation, and the minimum stable tests
  needed to protect public behavior.
* Every committed test must name a stable public contract or regression that it protects. Keep it deterministic, small, offline,
  and independent of private models, datasets, hardware, and experiment artifacts.
* Do not commit temporary test scripts, probes, experiment matrices, hardware runners, benchmark harnesses, debugging captures,
  generated outputs, or one-off validation code. Keep them in the external Task Contract artifact root.
* Do not commit experiment records as examples, including model-identity JSON files containing local paths, checkpoint or
  normalization hashes, machine-specific settings, or accepted-asset identities.
* Examples must demonstrate a reusable public API. They must not encode one internal experiment, machine, checkpoint package, or
  evaluation result.
* Keep datasets, checkpoints, ONNX files, engines, captures, traces, logs, videos, credentials, and large generated evidence
  outside Git.
* Before staging, audit the complete diff and remove files that do not satisfy this boundary. Stage explicit paths; do not use an
  unaudited `git add .`.

# Coding Style

## Fail Fast

* Prefer fail-fast over defensive programming.
* Do not add unnecessary validation, fallback paths, retries, or recovery logic.
* Assume inputs satisfy documented contracts unless there is a strong reason not to.
* Let errors surface early and visibly instead of hiding them.
* If something is wrong, crash with a clear error rather than continuing with potentially invalid state.

## Error Handling

* Do not use `try/catch` unless there is a specific and justified recovery strategy.
* Do not swallow exceptions.
* Do not add logging-only `catch` blocks.
* Prefer allowing exceptions to propagate to the top level.
* Recovery code should be rare and explicit.

## Code Style

* Keep code short, simple, and readable.
* Prefer straightforward implementations over highly abstract or configurable designs.
* Avoid excessive indirection, wrappers, and helper functions.
* Minimize boilerplate.
* Optimize for maintainability and clarity.

## Formatting

* We use wide screens.
* Prefer longer lines and fewer line breaks.
* Target a maximum line length of approximately 140 characters.
* Avoid wrapping expressions, function calls, and argument lists unless readability clearly improves.

## General Principle

When choosing between defensive and simple code, prefer the simpler solution. We value correctness, clarity, and fast failure over
robustness against impossible states.
