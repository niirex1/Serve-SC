.PHONY: install demo test data clean
install:
	python -m pip install -r requirements.txt
demo:
	python scripts/run_demo.py
data:
	python scripts/make_synthetic_data.py
test:
	python -m pytest -q
clean:
	rm -rf results/*.json results/*.md data/synthetic __pycache__ */__pycache__ .pytest_cache
