import re

with open("index.md", "r") as f:
    text = f.read()

urls = re.findall(r"http://121.199.45.168:[^\)]+", text)

import os

os.chdir("../img/dl")

for url in urls:
    os.system(f"wget {url}")