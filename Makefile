PY := .venv/bin/python

# Session eval — the deployment-shaped harness that produces the headline
# numbers in README/paper/results/session_summary.json (mirrors README's
# repro command). This is the default `make bench`.
bench:
	$(PY) -m majalis.bench.session --arms single,majalis,mad --seeds 0,1,2,3,4

# Older per-family harness (bench/run.py) — its outputs don't feed the
# current README/paper narrative; kept for the churn/multihop family
# ablations it still runs.
bench-families:
	$(PY) -m majalis.bench.run --arms single,sc5,mad,majalis --families churn,multihop --n 50 --seed 0

bench-smoke:
	$(PY) -m majalis.bench.run --arms single --families churn --n 3 --seed 0

test:
	$(PY) -m pytest tests/ -q

smoke:
	$(PY) scripts/smoke_test.py

# Rebuild the static demo and publish it into the Pages site (docs/demo/).
# No --live-backend: the published pages are pure replay and cannot bill.
site:
	$(PY) scripts/build_dashboard.py
	$(PY) scripts/build_live.py
	$(PY) scripts/translate_zh.py
	cp dashboard/index.html dashboard/index.zh.html dashboard/live.html dashboard/live.zh.html docs/demo/
