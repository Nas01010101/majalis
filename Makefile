PY := .venv/bin/python

bench:
	$(PY) -m agora.bench.run --arms single,sc5,mad,agora --families churn,multihop --n 50 --seed 0

bench-smoke:
	$(PY) -m agora.bench.run --arms single --families churn --n 3 --seed 0

test:
	$(PY) -m pytest tests/ -q

smoke:
	$(PY) scripts/smoke_test.py
