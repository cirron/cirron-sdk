# Python style guide

We follow the [Google Python Style Guide §3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
`ruff` enforces the mechanical half (`D` rules, `convention = "google"`); the
rest is review.

## Prose

Complete sentences with proper punctuation, in docstrings and comments alike.
Comments should read as narrative text rather than as fragments.

## Docstrings

- Every module, public class, and public function has one. The summary is a
  single sentence on the line with the opening `"""`, followed by a blank line
  if there is more (`D205`, `D212`).
- **Imperative mood.** Write "Return the resolved path", not "Returns the
  resolved path". Google permits either but demands consistency, and this
  codebase is overwhelmingly imperative (`Return` 88, `Open` 17, `Wrap` 15,
  `Close` 14). Properties are the exception: document them as the value they
  hold ("Number of scopes open on the calling thread"), not as an action.
- Omit on: `__init__` (document constructor parameters in the **class**
  docstring's `Args:`), obvious dunders (`__len__`, `__repr__`, `__enter__`),
  private helpers whose name and signature say everything, and `@overload` stubs.
- Sections in order: `Args:` / `Returns:` (or `Yields:`) / `Raises:`. Public
  classes with public annotated fields also get `Attributes:`, formatted like
  `Args:`.
- **Drop the type when it duplicates the annotation.** Write
  `name: Dataset name or scheme URI.`, not `name (str): Dataset name...`. Keep
  the `(type)` fragment only where it adds what the signature does not, meaning
  the parameter is annotated `Any` or `object`, or is unannotated. `ruff` cannot
  check this; reviewers can.
- **One canonical full docstring per public name**, on the outermost public
  re-export. That is `cirron/__init__.py` for the `ci.*` surface, since it is
  what `help()` and IDE hover resolve to. Layers below it get a summary plus
  ``Mirrors :func:`cirron.<name>`.`` and no `Args:`/`Returns:`/`Raises:` block.
  Never copy a parameter table into two files.
- `Examples:` uses a plain indented code block. **No `>>>` prompts**, because we
  do not run doctests and a prompt CI never executes will rot.
- Prose wraps at 88 columns (`W505`). House markup: ``double backticks`` for
  literals, `:func:` / `:class:` / `:mod:` / `:data:` for cross-references.

## Comments

- Write the *why*. If a comment restates the code, delete it.
- **Hard cap: four lines per standalone comment block.** If the rationale will
  not compress to four lines it is design documentation. Put it in the module
  docstring or in `docs/`.
- Banned: changelog voice ("previously", "this used to", "unchanged from round
  3"), ticket references, dated claims ("at the time of writing"), banner
  dividers, and `TODO` / `FIXME` / `XXX`. File an issue instead.
- Keep: measured results, **especially negative ones** ("`var_mean` was tried
  and measured slower"), correctness invariants, ordering and locking
  constraints, upstream-bug workarounds, and anything a reader would otherwise
  "fix" and thereby break.
- **Never document a rejected approach as though it shipped.** If you try
  something and back it out, the comment records the negative result in one
  line and no docstring mentions it.

## Tests

Test docstrings are exempt from `D` (`per-file-ignores`), per Google §3.8.2. The
test *name* is the spec, so make it long and descriptive. Add a docstring only
for rationale a reader cannot recover from the code: why this ordering, why this
test is not vacuous, what a naive version would fail to prove. Those are the most
valuable prose in the repo.

The prose and comment rules above apply to tests unchanged.

## Enabled rules and why

`pyproject.toml` selects `E, F, I, UP, B, W, A, D, RUF100`. The non-obvious
entries:

| Setting | Why |
|---|---|
| `convention = "google"` | Auto-disables the rules that conflict with Google format: `D203`, `D213`, `D400`, `D401`, `D406`-`D409`, `D413`. |
| `ignore = ["D107"]` | Constructor parameters are documented in the class docstring's `Args:` (Google §3.8.4). `D107` wants a second docstring on `__init__`, which is the wrong location. |
| `ignore = ["D105"]` | `__len__`, `__repr__`, `__enter__` are self-describing. |
| `ignore = ["D203", "D213"]` | Redundant under the google convention; pinned so the house choice (`D211`, `D212`) survives a convention change. |
| `ignore = ["E501"]` | Code-line length is `ruff-format`'s job. Doc and comment lines are governed by `W505` instead, which is a separate check. |
| `max-doc-length = 88` | Measured: 88 costs 9 rewraps repo-wide, 79 costs 111, and 72 costs over 1,000. |
| `"tests/**" = ["D"]` | Google §3.8.2 exempts test modules. boto3, httpx, and pydantic all do the same. |
| `RUF100` | Dead `# noqa` directives are a lie about what the code needs. Sixteen had accumulated before this was enabled. |

`D401` (imperative mood) stays off deliberately, even though imperative is the
house voice. It has 24 hits and most are properties, where a noun phrase is the
correct form. `D401` would force "Return the number of open scopes" onto a
property whose docstring should read "Number of scopes open on the calling
thread". The rule cannot tell the two cases apart, so mood is a review concern
here rather than a lint one.
