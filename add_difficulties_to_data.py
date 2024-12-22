# %%
import pandas as pd

data = pd.read_csv("data/merged_df.csv")
meta_data = pd.read_csv("data/ensembled_annotations_public.csv")
pass_rate_data = pd.read_csv("data/merged_df_pass_rate.csv")
print(meta_data.head())

# %%
print(len(meta_data))
print(meta_data.columns)
# %%
ids = data["id"]
print(ids)
# %%

subset_meta_data = meta_data[meta_data["instance_id"].isin(ids)]
print(subset_meta_data)

# %%

difficulties = subset_meta_data["difficulty"]
print(difficulties)

print(difficulties.value_counts())

# percentages of value counts
print(difficulties.value_counts() / len(difficulties))

# %%

# Ensure the difficulties are added to the dataset with matching ids
data_with_difficulties = data.copy()
data_with_difficulties["difficulty"] = data_with_difficulties["id"].map(subset_meta_data.set_index("instance_id")["difficulty"])

# Save the updated dataset to a new CSV file
data_with_difficulties.to_csv("data/merged_df_with_difficulties.csv", index=False)

print(data_with_difficulties.head())

# %%

# repeat the process for the pass_rate data

pass_rate_data_with_difficulties = pass_rate_data.copy()
pass_rate_data_with_difficulties["difficulty"] = pass_rate_data_with_difficulties["id"].map(subset_meta_data.set_index("instance_id")["difficulty"])

pass_rate_data_with_difficulties.to_csv("data/merged_df_pass_rate_with_difficulties.csv", index=False)

# %%
