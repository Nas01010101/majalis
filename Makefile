PY := .venv/bin/python

bench:
	$(PY) -m majalis.bench.run --arms single,sc5,mad,majalis --families churn,multihop --n 50 --seed 0

bench-smoke:
	$(PY) -m majalis.bench.run --arms single --families churn --n 3 --seed 0

test:
	$(PY) -m pytest tests/ -q

smoke:
	$(PY) scripts/smoke_test.py
