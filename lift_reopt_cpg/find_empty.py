import os
import pandas as pd

path = "Binaries/Dataset-2"

count = 0
empty = 0

result = {"arch": [], "project": [], "opt_level": [], "file_name": [], "c_dot": []}

def info_extract(filename: str):
    filename = filename.replace("c_dot_", "")
    arch, project_opt_level, file_name = filename.split("_")
    opt_level = project_opt_level[-2:]
    project = project_opt_level[:-3]
    return arch, project, opt_level, file_name


for c_dot_folder in os.listdir(path):
    c_dot_folder_path = os.path.join(path, c_dot_folder)
    if not os.path.isdir(c_dot_folder_path):
        continue
    if not c_dot_folder.startswith("c_dot_") or c_dot_folder == "c_dot_out" or c_dot_folder == "c_dot_workspace":
        continue
    if c_dot_folder.endswith(".tmp"):
        continue
    # if it is an empty folder
    if not os.listdir(c_dot_folder_path):
        empty += 1
        arch, project, opt_level, file_name = info_extract(c_dot_folder)
        result["arch"].append(arch)
        result["project"].append(project)
        result["opt_level"].append(opt_level)
        result["file_name"].append(file_name)
        result['c_dot'].append(c_dot_folder)
    count += 1
        
print("Total: {}, Empty: {}".format(count, empty))

df = pd.DataFrame(result)
df.to_csv("empty.csv", index=False)