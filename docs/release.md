# Release And Compatibility Policy

## Versioning

`pi-quant` follows semantic versioning for its Python API and CLI. Contract
`schema_version` values are independent from the package version. The 1.0.0
package keeps existing records at `schema_version=1`; it does not silently
rewrite v0.1-v0.5 YAML or JSON.

Within the 1.x line:

- additions must be backward compatible;
- new contract fields require defaults unless introduced in a new contract;
- fields, enum values, commands, and top-level public imports are not removed
  or reinterpreted;
- an incompatible contract needs a new schema version plus an explicit old
  reader or migration command;
- an incompatible Python or CLI change requires a new major package version.

The supported top-level import surface is `piquant.__all__`. Narrow adapter,
backend, compiler, evaluator, and store protocols remain available from their
documented modules. This policy does not create an automatic plugin registry;
callers continue to inject implementations explicitly. The wheel includes a
PEP 561 `py.typed` marker so external type checkers can consume the inline
annotations.

## Supported Environment

The default 1.0.0 package supports CPython 3.10 through 3.14 and contains only
NumPy, Pydantic, and PyYAML runtime dependencies. CI runs the default offline
suite and package build on every declared Python minor version.

Optional integrations remain explicit:

- ModelOpt is pinned to the public `nvidia-modelopt==0.45.0` integration;
- ONNX and ONNX Runtime use the ranges declared by the `onnx` extra;
- Pi0.5 data helpers use the packages declared by the `pi05` extra;
- OpenPI, FastWAM, TensorRT, CUDA, pi.cpp, datasets, and simulators are supplied
  and versioned by their target environments.

Default import, `piquant --version`, `doctor`, plan validation, search recipe
generation, ranking, promotion planning, and lineage validation must not import
optional runtimes. Selecting an unavailable integration fails explicitly.

## Release Gates

Before a v1.x feature PR is ready for human review:

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

The release workflow also installs the generated wheel into a fresh virtual
environment and runs:

```bash
piquant --version
piquant doctor
python -c "import piquant; assert piquant.__version__ == '1.0.0'"
```

Every committed recipe must validate through its existing public reader. The
complete diff must satisfy `AGENTS.md`: no model identity records, experiments,
hardware runners, ONNX, engines, captures, traces, logs, or generated evidence
may enter the PR.

## Artifact Lineage

Production handoffs should resolve and persist one
`ArtifactLineageManifest`. The manifest records immutable artifact hashes,
direct parent IDs, terminal nodes, the furthest evidence boundary, and a
canonical lineage hash. Use `piquant validate-lineage --check-artifacts` when
all referenced files are locally available.

A valid lineage proves identity consistency only through its declared boundary.
It does not turn source parity into target parity, standalone timing into
server/client timing, or closed-loop evidence into human acceptance.

## Merge, Tag, And Publication

A feature PR may become Ready after all blocking machine gates pass. The
maintainer separately decides whether to merge it. After merge, creating a
`v1.0.0` tag, creating a GitHub release, and publishing a package each require
explicit human authorization and verification that the tag points to the
accepted merge commit.

Library release acceptance never accepts a Pi0.5, FastWAM, AGX, RTX, pi.cpp, or
closed-loop deployment candidate. Candidate promotion retains its own lineage,
target, server/client, task, and human gates.
