# Living Taxonomy

A category map of arXiv cs.AI that revises itself daily. A GitHub Action pulls the newest abstracts, files them into last run's categories, founds new categories from strays, merges converging ones, retires idle ones, and revives retired ones when their topic returns.

Outputs, committed back to this repo each run:
- `data/tree.json`: current state
- `data/history/<date>.json`: daily snapshot of category names and sizes
- `data/stability.csv`: one row per run (categories, merged, retired, revived, churn)

Run once by hand: Actions tab, daily-taxonomy, Run workflow.
