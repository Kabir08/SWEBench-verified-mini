# %%
import pandas as pd

df = pd.read_csv("data/output.csv")
print(df.columns)
df.head()
# %%

df["IMAGE ID"] = df["IMAGE"].astype(str)
df["CREATED"] = df["ID"].astype(str) + " " + df["CREATED"].astype(str) + " " + df["SIZE"].astype(str)
df["SIZE"] = df["SHARED"].astype(str)
df["SHARED SIZE"] = df["SIZE.1"].astype(str)
df["UNIQUE SIZE"] = df["UNIQUE"].astype(str)
df["CONTAINERS"] = df["SIZE.2"].astype(str)

# Drop the redundant columns
df = df.drop(columns=['IMAGE', 'ID', 'SHARED', 'SIZE.1', 'UNIQUE', 'SIZE.2'])

# Reorder columns to match the expected format
correct_order = ["REPOSITORY", "TAG", "IMAGE ID", "CREATED", "SIZE", "SHARED SIZE", "UNIQUE SIZE", "CONTAINERS"]
df = df[correct_order]

df["ID"] = df["REPOSITORY"].str.replace("sweb.eval.x86_64.", "", regex=False)

# Convert sizes to GB
def convert_to_gb(size_str: str) -> float:
    """
    Convert a size string (GB, MB, or kB) to GB float value.
    
    Args:
        size_str: String containing size with unit (e.g., "77.9MB", "2.58GB", "500kB")
        
    Returns:
        Size in GB as float
    """
    size_str = size_str.strip()
    if 'GB' in size_str:
        return float(size_str.replace('GB', ''))
    elif 'MB' in size_str:
        return float(size_str.replace('MB', '')) / 1024
    elif 'kB' in size_str:
        return float(size_str.replace('kB', '')) / (1024 * 1024)
    else:
        return float(size_str)

df["SIZE_IN_GB"] = df["SIZE"].apply(convert_to_gb)

# %%
# map to the environment

import json
from pathlib import Path

# Load the JSON file
json_path = Path("data/instance_to_env.json")
with json_path.open("r") as file:
    instance_to_env = json.load(file)

# Map the ID to the environment using the instance_to_env dictionary
df["ENVIRONMENT"] = df["ID"].map(instance_to_env)


# %%
# Reset the index
df = df.reset_index(drop=True)

print("\nProcessed DataFrame:")
print(df.head())
# %%

df.to_csv("data/docker_image_sizes.csv", index=False)



# %%
