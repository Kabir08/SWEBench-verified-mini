# %%

import datasets
import json
# First load the original dataset
dataset = datasets.load_dataset("princeton-nlp/SWE-bench_Verified")


# %%
print(dataset)


# %%
representative_sample_ids = json.load(open("data/representative_sample_ids.json"))
print(len(representative_sample_ids))
print(representative_sample_ids)

# First check the IDs
print(f"Number of IDs in representative_sample_ids: {len(representative_sample_ids)}")
print(f"Number of unique IDs in representative_sample_ids: {len(set(representative_sample_ids))}")

# %% 

# Filter the dataset to only include rows where the ids match the representative sample ids
def filter_dataset_by_ids(dataset: datasets.DatasetDict, ids: list[str]) -> datasets.DatasetDict:
    """
    Filter the dataset to only include rows where the ids match the given list of ids.

    Parameters:
    - dataset: The original HuggingFace dataset.
    - ids: A list of ids to filter the dataset by.

    Returns:
    - A new HuggingFace dataset containing only the rows with the matching ids.
    """
    def is_in_ids(example):
        return example['instance_id'] in ids

    filtered_dataset = dataset.filter(is_in_ids)
    return filtered_dataset

# Apply the filter function to the dataset
filtered_dataset = filter_dataset_by_ids(dataset, representative_sample_ids)
print(f"Number of rows in filtered dataset: {len(filtered_dataset)}")

# %%
# Change the dataset name in the filtered dataset object
filtered_dataset._info.builder_name = "swe-bench-verified-tiny"

# %%
print(filtered_dataset)
def dataset_glimpse(dataset: datasets.DatasetDict, num_rows: int = 5) -> None:
    """
    Print a glimpse of the dataset and some metadata like the number of rows.

    Parameters:
    - dataset: The HuggingFace dataset.
    - num_rows: The number of rows to display as a glimpse.
    """
    # Print the number of rows in the dataset
    print(f"Number of rows in the dataset: {dataset.num_rows}")

    # Print a glimpse of the dataset
    for split, data in dataset.items():
        print(f"\nGlimpse of the '{split}' split:")
        print(data.to_pandas().head(num_rows))

# Show a glimpse of the filtered dataset
dataset_glimpse(filtered_dataset)


# %%
# Save the filtered dataset to a new directory
filtered_dataset.save_to_disk("data/filtered_huggingface_dataset")


# %%

# Then push it to your account as private
filtered_dataset.push_to_hub("MariusHobbhahn/swe-bench-verified-tiny", private=True)

# %%
