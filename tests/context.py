import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_folder   = os.path.join(project_root, 'src')

sys.path.insert(0, src_folder)
sys.path.insert(0, project_root)

import src