.PHONY: install test smoke uci archive
install:
	python -m pip install -e .
test:
	pytest -q
smoke:
	python scripts/run_experiment.py --config configs/uci.yaml --smoke-test
uci:
	bash scripts/reproduce_uci.sh
archive:
	git archive --format=zip --output=credfuzz-hgt-source.zip HEAD
