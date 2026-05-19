PYTHON := .venv/bin/python

all: figures main.pdf

main.pdf: main.tex sections/*.tex figures/*.pdf bibliography.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf

figures:
	cd .. && $(PYTHON) paper/figures/src/fig1_framework.py
	cd .. && $(PYTHON) paper/figures/src/fig2_metric_decomposition.py
	cd .. && $(PYTHON) paper/figures/src/fig3_moa_ceiling.py
	cd .. && $(PYTHON) paper/figures/src/fig4_driver_mutations.py
	cd .. && $(PYTHON) paper/figures/src/fig5_response_matching.py
	cd .. && $(PYTHON) paper/figures/src/fig6_generalization.py
	cd .. && $(PYTHON) paper/figures/src/sfig1_top_features.py
	cd .. && $(PYTHON) paper/figures/src/sfig2_perdrug_pathway.py

.PHONY: all clean figures
