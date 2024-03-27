#!/bin/bash

echo "Run handin 2"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

# echo "Downloading Dataset"
# if [ ! -e Vandermonde.txt ]; then
#   wget home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt
# fi

# echo "Run the script for 1a"
# python3 Poisson.py

# echo "Run the script for 2"
# python3 vandermonde.py



# code that makes a movie of the movie frames
#ffmpeg -framerate 25 -pattern_type glob -i "plots/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 sinemovie.mp4

echo "Generating the pdf"

pdflatex NUR-2.tex
#bibtex template.aux
#pdflatex template.tex
#pdflatex template.tex


