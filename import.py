import mysql.connector as connector
import os
import config
import pandas as pd
import numpy as np 
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import math

from dotenv import load_dotenv
from IPython.display import display  
from sklearn.feature_extraction.text import CountVectorizer

load_dotenv()

# Configuration for database connection
db_conn = {
    'user': config.DB_USER,
    'password': config.DB_PW,
    'host': config.DB_HOST,
    'database': 'tawos',
    'raise_on_warnings': True
}

cnx = connector.connect(**db_conn)
print("Connected to database.")

cur = cnx.cursor()

cur.execute("SET SESSION group_concat_max_len = 10000000;")

query = """SELECT 
    issue.ID AS Issue_ID, 
    issue.Priority, 
    issue.Creation_Date, 
    issue.Resolution_Date, 
    issue.Estimation_Date,
    issue.Story_Point,
    issue.Resolution,
    issue.Status,
    issue.Description,
    issue.Type,
    issue.Assignee_ID,
    p.ID AS Project_ID,
    p.Name AS Project_Name,
    p.Description AS Project_Description,
    s.ID AS Sprint_ID, 
    s.Name AS Sprint_Name, 
    s.State AS Sprint_State,
    r.ID AS Repository_ID,
    r.Name AS Repository_Name
FROM issue 
LEFT JOIN project p ON p.ID = issue.Project_ID 
LEFT JOIN sprint s ON s.ID = issue.Sprint_ID 
LEFT JOIN repository r ON p.Repository_ID = r.ID
WHERE issue.Sprint_ID IS NOT NULL AND issue.Resolution_Date IS NOT NULL AND issue.Resolution IS NOT NULL AND
issue.Resolution IN ('Complete', 'Fixed', 'Done', 'Implemented', 'Resolved', 'Deployed', 'Completed') AND issue.Status NOT IN
('Invalid', 'Won''t Fix')
"""


cur.execute(query)

df = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])

# Fixing the issue with indexing after groupby and apply
# grouped = df.groupby(['Project_ID', 'Sprint_Name'])

# only including RESOLVED linked issues (so must be a part of the 31,000)

priority_mapping = {
    'Blocker': 1,
    'Blocker - P1': 1,
    'Highest': 1,
    
    'Critical': 2,
    'Critical - P2': 2,
    'High': 2,
    
    'Major': 3,
    'Major - P3': 3,
    'Medium': 3,
    
    'Minor': 4,
    'Minor - P4': 4, 
    'Low': 4,
    
    'Trivial': 5,
    'Trivial - P5': 5,
    'Lowest': 5,
}

df["Priority_Class"] = df["Priority"].map(priority_mapping)

df["Story_Point_Norm"] = df["Story_Point"].replace(0, np.nan)
df.loc[df["Story_Point_Norm"].notna(), "Story_Point_Norm"] = (df.loc[df["Story_Point_Norm"].notna(), "Story_Point_Norm"]
      .round()
      .clip(lower=1, upper=10)
)

# Remove issues resolved in a suspiciously quick time that is high priority or high effort
df['Creation_Date'] = pd.to_datetime(df['Creation_Date'])
df['Resolution_Date'] = pd.to_datetime(df['Resolution_Date'])
# remove items created then immediately resolved only if unlikely
df['Resolution_Duration_Hours'] = (df['Resolution_Date'] - df['Creation_Date']).dt.total_seconds() / 3600
df["Issue_ID"] = df["Issue_ID"].astype(int)

df = df[~((df['Resolution_Duration_Hours'] <= 1) & ((df['Priority_Class'].isin([1, 2, 3])) | (df['Story_Point_Norm'] >= 5)))]

# 201470 valid issues but most arent a part of sprints... (37261) if only including ones in sprints
# 194708 valid issues after filter for retroactively created issues (36771) if only including ones in sprints - might adjust because high priority doesnt necessarily mean it takes long

query = """SELECT 
    Issue_ID,
    Target_Issue_ID,
    Name,
    Description,
    Direction
FROM Issue_Link
WHERE Issue_ID IS NOT NULL
"""
cur.execute(query)

linked_df = pd.DataFrame(cur.fetchall(), columns=[i[0] for i in cur.description])
linked_df["Issue_ID"] = linked_df["Issue_ID"].astype(int)
linked_df["Target_Issue_ID"] = linked_df["Target_Issue_ID"].astype(int)
linked_df["Name"] = linked_df["Name"].str.title()
print("start", linked_df.shape)

linked_df = linked_df[(linked_df["Name"].str.contains("Gantt|Depend|Block|Required|Complete|Child|Parent|Follow", case=False, na=False)) & (linked_df["Name"] != "Multi-Level Hierarchy [Gantt]") & (~linked_df["Description"].str.contains("together"))]
print("name filter", linked_df.shape)

def normalize_dependency_relation(relation):
    relation = relation.lower()

    # These imply the issue DEPENDS ON another issue
    depends_on_keywords = [
        'depends on', 'has to be done after', 'is blocked by', 'requires', 'blocked', 'is fixed by', 'depends upon', 'dependent', 'start is earliest end of',
        'FS-depends on', 'SS-depends on', 'FF-depends on', 'is dependent of', 'follows'
    ]

    # These imply the issue IS DEPENDED ON BY another issue
    depended_on_by_keywords = [
       'is depended on by', 'has to be done before', 'blocks', 'fixes', 'is depended upon by', 'is required by', 'earliest end is start of', 'required by', 'is FS-depended by',
       'is FF-depended by', 'is SS-depended by', 'followed by', 'depended on by'
    ]

    if relation in depends_on_keywords:
        return "depends on"
    elif relation in depended_on_by_keywords:
        return "is depended on by"

    
linked_df["Description_Norm"] = linked_df["Description"].apply(normalize_dependency_relation)
linked_df = linked_df.dropna(subset=["Description_Norm"])
print("description norm filter", linked_df.shape)

linked_df[["Final_Issue", "Initial_Issue"]] = linked_df.apply(lambda x: pd.Series([x.Issue_ID, x.Target_Issue_ID]) if x.Description_Norm == "depends on" else pd.Series([x.Target_Issue_ID, x.Issue_ID]) ,axis=1)

linked_df[["Issue_ID", "Target_Issue_ID", "Name", "Description", "Description_Norm", "Initial_Issue", "Final_Issue"]]
print("double check filter", linked_df.shape)

linked_df = linked_df.drop_duplicates(subset=["Initial_Issue", "Final_Issue"], keep="first")
print("drop duplicates", linked_df.shape)

all_valid_issues = df["Issue_ID"].unique().tolist()
linked_df = linked_df[(linked_df["Initial_Issue"].isin(all_valid_issues)) & (linked_df["Final_Issue"].isin(all_valid_issues))]
print("valid issues", linked_df.shape)

linked_df = linked_df.merge(df[["Issue_ID", "Resolution_Date"]], how="inner", on=["Issue_ID"])
print("issue id merge", linked_df.shape)
linked_df = linked_df.rename(columns={"Resolution_Date": "Issue_ID_Resolution_Date"})
print("double check issue id merge", linked_df.shape)

linked_df = linked_df.merge(df[["Issue_ID", "Resolution_Date"]], how="inner", left_on=["Target_Issue_ID"], right_on=["Issue_ID"])
print("target id merge", linked_df.shape)
linked_df = linked_df.rename(columns={"Resolution_Date":"Target_Issue_ID_Resolution_Date", "Issue_ID_x": "Issue_ID"})
print("double check target id merge", linked_df.shape)
linked_df = linked_df.drop(columns=["Issue_ID_y"])
print("triple check target id merge", linked_df.shape)

# remove contradictory dependencies
linked_df = linked_df[~((linked_df["Issue_ID_Resolution_Date"] < linked_df["Target_Issue_ID_Resolution_Date"]) & (linked_df["Description_Norm"] == "depends on"))]
print("remove conflicting 1", linked_df.shape)

linked_df = linked_df[~((linked_df["Issue_ID_Resolution_Date"] > linked_df["Target_Issue_ID_Resolution_Date"]) & (linked_df["Description_Norm"] == "is depended on by"))]
print("remove conflicting 2", linked_df.shape)


dependencies_df = linked_df.groupby('Final_Issue')['Initial_Issue'].agg(list).reset_index()
print("group by (dependencies df)", dependencies_df.shape)

# issue id is the issue and then dependencies are the ones issue id depends on
dependencies_df.rename(columns={'Final_Issue':'Issue_ID','Initial_Issue': 'Dependencies'}, inplace=True)
print("double check group by (dependencies df)", dependencies_df.shape)

df = df.merge(dependencies_df, how="left", on=["Issue_ID"])
df["Estimation_Date"] = pd.to_datetime(df["Estimation_Date"])
df["Creation_Date"] = pd.to_datetime(df["Creation_Date"]).dt.date
df["Resolution_Date"] = pd.to_datetime(df["Resolution_Date"])

# filter on top 5 sprints in specific projects
sprint_df = df[df["Project_ID"].isin(config.PROJECTS)]
# top 5 largest sprints BEFORE adding dependencies
sprint_df = sprint_df.groupby(['Project_ID', 'Sprint_ID']).size().reset_index(name='Issue_Count').sort_values(by=["Project_ID", "Issue_Count"], ascending=[True, False]).groupby("Project_ID").head(5).reset_index(drop=True)
sprint_df["Top"] = True

sprint_df = df.merge(sprint_df, how="inner", on=["Project_ID", "Sprint_ID"])
sprint_df = sprint_df.drop_duplicates(subset=["Issue_ID"], keep="first")

sprint_count = 0
current_proj = None 

# based on fastest resolution times overall -- differs slightly when analysing per project
type_mapping = {
    "Release": 1,
    "Sub-task": 2,
    "Story": 3,
    "Technical task": 4,
    "Investigation": 5,
    "Test Task": 6,
    "Enhancement Request": 7,
    "Improvement": 8,
    "Bug": 9,
    "Task": 10,
    "Suggestion": 11,
    "Support Request": 12,
    "New Feature": 13,
    "Technical Debt": 14,
    "Epic": 15,
    "Wish": 16,
    "Documentation": 17
}
    
# add type_class 
sprint_df["Type_Class"] = sprint_df["Type"].map(type_mapping)

if os.path.exists("corr"):
    shutil.rmtree("corr")
os.makedirs("corr", exist_ok=True)

sprint_id_to_filename = {}  # {(proj, sprint_id): sprint_count}

top_sprints = sprint_df[sprint_df["Top"] == True] \
    .groupby(['Project_ID', 'Sprint_ID']) \
    .size() \
    .reset_index(name='Issue_Count') \
    .sort_values(by=["Project_ID", "Issue_Count"], ascending=[True, False])

if os.path.exists("input"):
    shutil.rmtree("input")

for _, row in top_sprints.iterrows():
    proj = row["Project_ID"]
    sprint = row["Sprint_ID"]

    if proj != current_proj:
        current_proj = proj
        sprint_count = 1
    else:
        sprint_count += 1

    group = sprint_df[(sprint_df["Project_ID"] == proj) & (sprint_df["Sprint_ID"] == sprint)]
    # for all sprints except first one, carry over from previous
    if sprint_count > 1:
        p_rollover_df = rollover_df
        group = pd.concat([group, p_rollover_df], axis=0)

    deps = group["Dependencies"].explode().dropna().unique()

    extra_rows = df[df["Issue_ID"].isin(deps)].copy()

    # Wipe their dependencies so they can't sneak in extra
    extra_rows["Dependencies"] = np.nan

    group = pd.concat([group, extra_rows], axis=0).drop_duplicates(subset=["Issue_ID"], keep="first")
    
    sprint_id_to_filename[(proj, sprint)] = sprint_count

    vio_set = set()
    folder_name = f"P{proj}"
    folder_path = os.path.join("input", folder_name)

    os.makedirs(folder_path, exist_ok=True)

    filename = os.path.join(folder_path, f"S{sprint_count}.xlsx")
    print(f"sprint {proj}, {sprint_count}: {group.shape}")
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        group = group.drop_duplicates(subset=["Issue_ID", "Project_ID", "Sprint_ID"]).sort_values(by=["Resolution_Date"], ascending=True).reset_index(drop=True)
        group["Issue_Rank"] = group.index + 1

        # rollover df (this is what the current sprint is carrying over to next sprint)
        carry_over = math.ceil(len(group) * config.CARRY_OVER_PERCENTAGE / 100)
        print(f"sprint {proj}, {sprint_count}, carry: {carry_over}")
        rollover_df = group.tail(carry_over)
        group = group.drop(rollover_df.index)

        # info df   
        info_df = group
        info_df[["Issue_ID", "Project_ID", "Sprint_ID", "Assignee_ID", "Description", "Type", "Type_Class", "Resolution_Date", "Creation_Date", "Estimation_Date", "Issue_Rank", "Priority", "Priority_Class", "Story_Point", "Story_Point_Norm", "Dependencies"]].to_excel(writer, sheet_name='Info', index=False)

        # gs df
        gs_df = group
        gs_df[["Issue_ID", "Project_ID", "Sprint_ID", "Assignee_ID", "Issue_Rank"]].to_excel(writer, sheet_name='Gold_Standard', index=False)

        # priority df
        prio_df = group[~group["Priority_Class"].isna()]
        prio_df = prio_df.sort_values(by=["Priority_Class"], ascending=True) # lower priority class tackled first
        prio_df[["Issue_ID", "Project_ID", "Sprint_ID", "Assignee_ID", "Priority", "Priority_Class"]].to_excel(writer, sheet_name='Priorities', index=False)

        # dependency df
        dep_df = group[~group["Dependencies"].isna()]
        dep_df[["Issue_ID", "Project_ID", "Sprint_ID", "Assignee_ID", "Dependencies"]].to_excel(writer, sheet_name='Dependencies', index=False)

        # baseline df (naive approach) - update to be creation dates df
        creation_date_df = group.sort_values(by=["Creation_Date"], ascending=True).reset_index(drop=True)
        creation_date_df[["Issue_ID", "Project_ID", "Sprint_ID", "Assignee_ID", "Creation_Date"]].to_excel(writer, sheet_name='Creation_Date', index=False)

        type_df = group
        type_df = type_df.sort_values(by=["Type_Class"], ascending=True) # lower type class tackled first
        type_df[["Issue_ID", "Project_ID", "Sprint_ID", "Assignee_ID", "Type", "Type_Class"]].to_excel(writer, sheet_name='Issue_Type', index=False)

        # What was rolled over from previous sprint
        if sprint_count > 1:
            p_rollover_df[["Issue_ID", "Issue_Rank", "Project_ID", "Sprint_ID", "Assignee_ID", "Description", "Type", "Type_Class", "Resolution_Date", "Creation_Date", "Estimation_Date", "Priority", "Priority_Class", "Story_Point", "Story_Point_Norm", "Dependencies"]].to_excel(writer, sheet_name='Rollover', index=False)

        group["Creation_Date"] = pd.to_datetime(group["Creation_Date"])
        group["Resolution_Date"] = pd.to_datetime(group["Resolution_Date"])
        corr = group[group["Project_ID"] == proj][['Priority_Class', 'Creation_Date', 'Type_Class', 'Resolution_Date']].corr(method='spearman')
        sns.heatmap(corr, annot=True, cmap='coolwarm')

        plt.figure(figsize=(8, 6))  
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f'Correlations For P{proj}', fontsize=14)
        plt.xticks(rotation=45, ha='right')  
        plt.yticks(rotation=0)  
        plt.tight_layout()  
        plt.savefig(os.path.join('corr', f"proj{proj}_sprint_{sprint}.pdf"))
        plt.close()

for proj in config.PROJECTS:
    sprint_df["Creation_Date"] = pd.to_datetime(sprint_df["Creation_Date"])
    sprint_df["Resolution_Date"] = pd.to_datetime(sprint_df["Resolution_Date"])
    corr = sprint_df[sprint_df["Project_ID"] == proj][['Priority_Class', 'Creation_Date', 'Type_Class', 'Resolution_Date']].corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='coolwarm')

    plt.figure(figsize=(8, 6))  
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f'Correlations For P{proj}', fontsize=14)
    plt.xticks(rotation=45, ha='right')  
    plt.yticks(rotation=0)  
    plt.tight_layout()

    plt.savefig(os.path.join('corr', f"proj{proj}.pdf"))
    plt.close()

sprints = {"S1": [], "S2": [], "S3": [], "S4": [], "S5": []}
proj_list = []

for proj in os.listdir("input"):
    proj_list.append(proj)
    for sprint_filename in os.listdir(os.path.join("input", proj)):
        sprint, ext = os.path.splitext(sprint_filename)

        if sprint in sprints:
            loc = os.path.join("input", proj, sprint_filename)

            info_df = pd.read_excel(loc, sheet_name="Info")
            sprints[sprint].append(info_df.shape[0])


df = pd.DataFrame({
    "proj": proj_list, 
    "S1": sprints["S1"],
    "S2": sprints["S2"],
    "S3": sprints["S3"],
    "S4": sprints["S4"],
    "S5": sprints["S5"]
})

df.to_csv(os.path.join('corr', "proj_sprint_counts.csv"))

# import -> main