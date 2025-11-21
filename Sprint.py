import config
import os
import pandas as pd
import re

class Sprint:
    def __init__(self, id, proj_dir):
        self.id = id
        self.project_id = int(re.search(r"[\\/]+P(\d+)$", proj_dir).group(1))

        self.loc = os.path.join(proj_dir, f"S{self.id}.xlsx")
        
        self.info_df = pd.read_excel(self.loc, sheet_name="Info")
        self.gs_df = pd.read_excel(self.loc, sheet_name="Gold_Standard")
        self.prio_df = pd.read_excel(self.loc, sheet_name="Priorities")
        self.dep_df = pd.read_excel(self.loc, sheet_name="Dependencies")
        self.creation_date_df = pd.read_excel(self.loc, sheet_name="Creation_Date")
        self.type_df = pd.read_excel(self.loc, sheet_name="Issue_Type")

        self.issues = self.gs_df["Issue_ID"].unique().tolist()
        self.idx_dict = {issue: i for i, issue in enumerate(self.issues)}
        self.issue_dict = {i: issue for i, issue in enumerate(self.issues)}
    