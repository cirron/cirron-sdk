---
name: Bug report
about: Report a defect in the Cirron SDK
title: ''
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is: what you expected to happen and what actually happened.

**Minimal reproducer**
The smallest self-contained Python snippet that triggers the bug.

```python
import cirron as ci

ci.profile()
# ...
```

**Environment**
Preferred: paste the output of `ci.deps()`, which captures Python version and every relevant framework in one go:

```bash
uv run python -c "import cirron as ci, sys; print('python', sys.version); print(ci.deps())"
```

If `ci.deps()` itself crashes (it might, depending on the bug), paste this fallback instead:

```bash
python --version
pip list 2>/dev/null | grep -iE 'cirron|torch|tensorflow|transformers|pandas|polars|numpy'
```

- OS (e.g. macOS 14.5, Ubuntu 22.04, Windows 11):
- Cirron SDK version:
- Hardware (CPU only / NVIDIA GPU + CUDA version / Apple Silicon / TPU):

**Additional context**
Tracebacks, spool snippets (`./.cirron/spool/*.json`), or `ci.trace()` output that helps narrow it down.
