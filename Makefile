SHELL=/bin/bash
MAIN=main
LATEX=pdflatex
BIBTEX=bibtex
DTM=$(shell date +%Y%m%d-%H%M%S)
EXTRA_ARGS=-shell-escape

all: clean build clear

build:
	@echo $(EXTRA_ARGS)
	$(LATEX) $(EXTRA_ARGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(EXTRA_ARGS) $(MAIN).tex
	$(LATEX) $(EXTRA_ARGS) $(MAIN).tex

clean:
	@rm -f $(MAIN).{pdf,ps,log,lot,lof,toc,out,dvi,bbl,blg} *.aux
	@echo Cleared all temporary files and $(MAIN).pdf

clear:
	@rm -f $(MAIN).{ps,log,lot,lof,toc,out,dvi,bbl,blg} *.aux
	@echo Cleared all temporary files

draft:
	@cp $(MAIN).pdf $(MAIN)-draft-$(DTM).pdf
	@echo Saved the draft as: $(MAIN)-draft-$(DTM).pdf

