python3 slide_type_elf.py --path index.ipynb
jupyter nbconvert index.ipynb --to slides
mv index.slides.html index.html
open index.html