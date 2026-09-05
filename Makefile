PYTHON ?= python3
EXPORT ?= export

.DEFAULT_GOAL := help
.PHONY: help demo pipeline notebooks replay

help:
	@echo "make demo       Route the synthetic example; no dataset needed"
	@echo "make pipeline   Rebuild local results and the trained model (requires export/)"
	@echo "make notebooks  Regenerate saved notebooks from local results"
	@echo "make replay     Replay the original held-out examples (requires local results)"

demo:
	$(PYTHON) scripts/route_request.py examples/request.json

pipeline:
	$(PYTHON) scripts/load_trajectories.py "$(EXPORT)"
	$(PYTHON) scripts/audit_trajectory_groups.py "$(EXPORT)"
	$(PYTHON) scripts/build_feature_table.py "$(EXPORT)"
	$(PYTHON) scripts/build_evaluation_table.py "$(EXPORT)"
	$(PYTHON) scripts/build_manual_review_sample.py "$(EXPORT)"
	$(PYTHON) scripts/train_two_stage_router.py --export "$(EXPORT)"
	$(PYTHON) scripts/two_stage_router.py
	$(PYTHON) scripts/evaluate_two_stage_router.py --export "$(EXPORT)"

notebooks:
	$(PYTHON) scripts/build_analysis_notebooks.py

replay:
	$(PYTHON) scripts/demo_router.py
