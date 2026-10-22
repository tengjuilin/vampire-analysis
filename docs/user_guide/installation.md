# Installation

`vampire-analysis` needs **Python 3.13 or newer**. It depends only on the
scientific Python stack (NumPy, SciPy, pandas, Matplotlib, scikit-image,
scikit-learn), all installed automatically. The import name is `vampire`; by
convention it is imported as `vp`.

## With pip

Inside an activated virtual environment (recommended, so the package does not
conflict with other projects):

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install vampire-analysis
```

If you use Anaconda/Miniconda, create an environment with a recent Python
first (`conda create -n vampire python=3.13`, then `conda activate vampire`)
and run the same `pip install` command.

## With uv

[uv](https://docs.astral.sh/uv/) manages Python versions and environments for
you. To add `vampire-analysis` to an existing uv project:

```bash
uv add vampire-analysis
```

To try it without creating a project:

```bash
uv run --with vampire-analysis python -c "import vampire as vp; print(vp.__version__)"
```

## Check the installation

```python
import vampire as vp

print(vp.__version__)
images, filenames = vp.datasets.load_example_images()
print(len(images), images[0].shape)  # 2 (512, 512)
```

## Jupyter

The tutorials in this documentation are executable notebooks (MyST markdown).
To run your own analysis interactively, install JupyterLab into the same
environment (`pip install jupyterlab` or `uv add --dev jupyterlab`) and start
it with `jupyter lab`. Figures returned by `vampire.plotting` display inline.

## Development setup

To work on `vampire-analysis` itself, clone the repository and let uv create
the environment with all development dependency groups (tests, lint, docs):

```bash
git clone https://github.com/tengjuilin/vampire-analysis.git
cd vampire-analysis
uv sync                                   # creates .venv from uv.lock
uv run pytest                             # unit tests + doctests
uv run ruff check . && uv run ruff format --check .
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the built documentation. The
tutorials are executed during the build; any error fails it. Set
`NB_EXECUTION_MODE=cache` to reuse notebook outputs between local rebuilds.
See `AGENTS.md` in the repository for code conventions.
