# Compile LaTeX document
# Args:
#   $1 -- path to project directory
#   $2 -- TeX-source file name (optional, default: main.tex)

FILE=${2:-main.tex}

latexmk -c -cd "$1/$FILE"
latexmk -pdflatex='pdflatex -file-line-error -synctex=1 -interaction=nonstopmode' \
        -pdf \
        -cd \
        -outdir=build \
        "$1/$FILE"