#!/bin/bash

echo "Run handin 2"

if [ ! -d "plots" ]; then
  echo "Creating Plotting Directory!"
  mkdir plots
fi

if [ ! -d "output" ]; then
  echo "Creating Output Directory!"
  mkdir output
fi

if [ ! -e output/1a.txt ] || [ ! -e plots/my_solution_1c.png ] || [ ! -e plots/my_solution_1b.png ]; then
  echo "Run the script for exercise 1"
  python3 ex1.py
fi

if [ ! -e output/2a.txt ] || [ ! -e output/2b.txt ] || [ ! -e plots/2a.png ] || [ ! -e plots/2b.png ]; then
  echo "Run the script for exercise 2"
  python3 ex2.py
fi

echo "Generating the pdf"
pdflatex -interaction=nonstopmode NUR-2.tex > latex_output1.txt
#bibtex template.aux > bibtex_output.txt
pdflatex -interaction=nonstopmode NUR-2.tex > latex_output2.txt
pdflatex -interaction=nonstopmode NUR-2.tex > latex_output3.txt


