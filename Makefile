.PHONY: all install test experiments exp1 exp2 exp3 exp4 plot serve simulate clean

PYTHON ?= python

all: test experiments plot

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v -p no:cacheprovider

experiments: exp1 exp2 exp3 exp4

exp1:
	$(PYTHON) experiments/exp1_contraction_breakdown.py

exp2:
	$(PYTHON) experiments/exp2_phase_diagram.py

exp3:
	$(PYTHON) experiments/exp3_duopoly_polarization.py

exp4:
	$(PYTHON) experiments/exp4_ablation_exploration.py

plot:
	$(PYTHON) visualization/plot_figures.py

serve:
	$(PYTHON) app.py

simulate:
	$(PYTHON) -m src.cli simulate

clean:
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__ experiments/__pycache__ visualization/__pycache__ tests/__pycache__ .pytest_cache
