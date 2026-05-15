"""Notebook-style quickstart for cell_classifier.

Run inside an editable install: `pip install -e .` from the project root.
Then either open this file in Jupyter (it's a .py percent-format you can
convert to .ipynb via jupytext or VSCode), or just `python quickstart.py`
to execute.
"""

# %% [markdown]
# # cell_classifier — quickstart
#
# Load the latest preprocess bundle and inspect class balance + feature stats.

# %%
from cell_classifier.data.loader import load_dataset

ds = load_dataset(N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2")
print(f"loaded {len(ds)} cells with {len(ds.feature_names)} features")
print(f"trainable: {int(ds.label_mask.sum())}")
print(f"  pass: {int(ds.y[ds.label_mask].sum())}")
print(f"  bad:  {int((1 - ds.y[ds.label_mask]).sum())}")

# %% [markdown]
# Discover existing validation runs.

# %%
from cell_classifier.utils.discover import find_runs

runs = find_runs(mode="validation")
for r in runs:
    print(f"  {r['slug']}  →  {r['path']}")
