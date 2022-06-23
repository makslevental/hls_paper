set -e

pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex


# latexmk --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=. main