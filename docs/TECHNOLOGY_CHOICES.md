# Technology choices and justification

| Area | Choice | Why | Alternatives considered |
|---|---|---|---|
| Language | Python 3.10+ | De facto standard in scientific computing and machine learning; the same language as typical model-training code, so the module can be dropped into an experiment pipeline. | R (strong statistics, weaker deep-learning ecosystem); Julia (fast, smaller community). |
| Numerics | NumPy | Vectorised array maths; binning is done with `digitize`/`bincount`, with no Python loops over samples. | Pure Python (too slow, more code); pandas (heavier than needed). |
| Optimisation | SciPy (`minimize_scalar`, `expit`, `logit`) | Robust bounded 1-D optimiser and numerically stable sigmoid/logit. Temperature is optimised as log T, which keeps T positive. | Hand-written gradient descent (more code, more bugs); PyTorch (large dependency for one scalar). |
| Plotting | Matplotlib (optional extra) | Standard for publication-quality figures; kept optional so the core stays light. | Plotly/Seaborn (extra dependencies, no benefit here). |
| Packaging | `pyproject.toml` + setuptools, `src/` layout | Modern PEP 517/621 packaging; `src/` layout prevents accidentally importing the working copy instead of the installed package. | Poetry/Hatch (fine, but add tooling to learn); `setup.py` (legacy). |
| Testing | pytest + pytest-cov | Concise tests, parametrisation, fixtures (`tmp_path`), coverage report. Tests check known analytic values (e.g. ECE of an always-0.9 model that is right half the time = 0.4) and statistical properties (recovering a known temperature). | `unittest` (more boilerplate). |
| Linting | Ruff | One fast tool replaces flake8, isort and pyupgrade; enforces consistent style in CI. | flake8 + black + isort (three tools). |
| Version control | Git, hosted on GitHub | Distributed history, branching, pull requests, issues, and native CI integration. | GitLab, Bitbucket. |
| CI/CD | GitHub Actions | Lives next to the code, free for public repositories, matrix builds over Python versions, marketplace actions for release. | Jenkins (self-hosted maintenance); GitLab CI (needs GitLab). |
| License | MIT | Permissive and widely understood in academia; allows reuse in other research code. | Apache-2.0, GPL. |

## CI/CD pipeline

`.github/workflows/ci.yml` defines four jobs:

1. **lint**: `ruff check .`
2. **test**: matrix over Python 3.10, 3.11, 3.12, 3.13; installs the package with dev extras,
   runs pytest with a minimum coverage gate (`--cov-fail-under=90`) and smoke-tests the demo.
3. **build**: after lint and tests pass, builds sdist and wheel and validates metadata with
   `twine check` (continuous *delivery* of an installable artifact).
4. **release**: only for tags matching `v*`; creates a GitHub Release with the built files.

## Design notes

- Inputs are validated up front (shape, 0/1 labels, probabilities in [0, 1], no NaN) so errors
  are explicit rather than silently wrong metrics.
- Metric functions are pure and stateless; the only stateful class is `TemperatureScaler`,
  following the familiar `fit` / `transform` convention of scikit-learn.
- Temperature scaling is monotone, so it does not change ranking metrics such as AUC.
