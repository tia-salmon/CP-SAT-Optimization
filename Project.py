import config
import os
import pandas as pd
import re
from Sprint import Sprint

class Project:
    def __init__(self, id):
        self.id = id
        self.base_dir = os.path.join(os.getcwd(), "input", f"P{id}") 
        self.sprints = [Sprint(int(re.search(r'S(\d+)\.xlsx', file).group(1)), self.base_dir) for file in os.listdir(self.base_dir) if os.path.isfile(os.path.join(self.base_dir, file))]
