import re
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str)

args = parser.parse_args()

with open(args.path,"r") as f:
    text = f.read()
    # add slide_type to metadata
    text = text.replace('"metadata": {}','''"metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
    }''')
    # remove id
    text = re.sub('\s+"id": ".+",',"",text)
    
with open(args.path,"w") as f:
    f.write(text)