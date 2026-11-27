#!/usr/bin/env bash
# Builds the JMIR Word manuscript and its Multimedia Appendix from ../main.tex.
# Needs: pandoc, python-docx, pillow.
set -e
cd "$(dirname "$0")"
python3 build.py >/dev/null
python3 assemble.py
cd build
for f in manuscript appendix_full; do pandoc $f.tex -f latex -t docx --shift-heading-level-by=1 -o $f.docx; done
cd ..
python3 post.py
cp build/JMIR_manuscript.docx JMIR_manuscript.docx
cp build/JMIR_Multimedia_Appendix_1.docx JMIR_Multimedia_Appendix_1.docx
echo "Built JMIR_manuscript.docx, JMIR_Multimedia_Appendix_1.docx and figures_upload/"
