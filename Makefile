# One pipeline: `make all` runs every stage from download to report.
# Each stage also has its own target. Later weeks append stages to STAGES.

PYTHON ?= .venv/bin/python
RUN := PYTHONPATH=src $(PYTHON) -m

STAGES := download parse audit

.PHONY: all setup $(STAGES) test clean

all: $(STAGES)

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

download:
	$(RUN) ipl_late_overs.download

parse: download
	$(RUN) ipl_late_overs.parse

audit: parse
	$(RUN) ipl_late_overs.audit

test:
	$(PYTHON) -m pytest

# Removes derived artefacts only. data/raw and reports/weekly are never touched.
clean:
	rm -rf data/interim/*.parquet data/processed/* models/* reports/data_audit.md reports/figures/*
	touch data/processed/.gitkeep models/.gitkeep reports/figures/.gitkeep
