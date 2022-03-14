import sys
import pathlib
parent = pathlib.Path.cwd().parent
sys.path.append(str(parent))
print(sys.path)
from evaluate import helper
