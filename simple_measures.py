# %% 
# use basic correlations and clustering techniques to find the best subset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#data = pd.read_csv("data/merged_df_with_difficulties.csv")
data = pd.read_csv("data/merged_df_pass_rate_with_difficulties.csv")

# %%

# replace the difficulty column with the mapped values
map_difficulty = {
    '<15 min fix': 0,
    '15 min - 1 hour': 1,
    '1-4 hours': 2,
    '>4 hours': 3
}

data["difficulty"] = data["difficulty"].map(map_difficulty)

# remove the id column and replace nans with 0
data_raw = data.drop(columns=["id"]).fillna(0)
data_raw.head()

# %%
nan_counts = data_raw.isna().sum()
print(nan_counts)

# %%

from sklearn.cluster import KMeans

def compute_kmeans(data: pd.DataFrame, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute k-means clustering on the given data.

    Parameters:
    - data: A pandas DataFrame containing the data to cluster.
    - n_clusters: The number of clusters to form.

    Returns:
    - A tuple containing the cluster labels and the cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# Example usage
n_clusters = 10
labels, centers = compute_kmeans(data_raw, n_clusters)
print("Cluster labels:", labels)
print("Cluster centers:", centers)

# %%

label_counts = np.bincount(labels)
print("Label counts:", label_counts)

# %%

data["cluster"] = labels
data_wo_id = data.drop(columns=["id"])
data_wo_id.head()

# %%
# Group the dataframe by cluster labels and compute the mean value for all other columns
grouped_data = data_wo_id.groupby("cluster").mean()
grouped_data.head(n=10)

# %%
from sklearn.utils import resample

def draw_representative_sample(data: pd.DataFrame, labels: np.ndarray, n: int, random_state: int = 1) -> pd.DataFrame:
    """
    Draw N representative datapoints from the data based on k-means clustering labels.

    Parameters:
    - data: A pandas DataFrame containing the data to sample from.
    - labels: A numpy array containing the cluster labels for each data point.
    - n: The number of representative datapoints to draw.

    Returns:
    - A pandas DataFrame containing the representative sample.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    
    # Calculate exact fractional samples
    exact_samples = proportions * n
    # Get initial counts and fractional parts
    sample_counts = exact_samples.astype(int)
    fractions = exact_samples - sample_counts
    
    # Distribute remaining samples based on largest fractional parts
    remaining = n - sample_counts.sum()
    if remaining > 0:
        # Get indices of largest fractional parts
        fraction_indices = np.argsort(fractions)[-remaining:]
        for idx in fraction_indices:
            sample_counts[idx] += 1
    
    representative_sample = pd.DataFrame()
    for label, count in zip(unique_labels, sample_counts):
        cluster_data = data[labels == label]
        count = min(count, len(cluster_data))
        cluster_sample = resample(cluster_data, n_samples=count, random_state=random_state, replace=False)
        representative_sample = pd.concat([representative_sample, cluster_sample])

    assert len(representative_sample) == n, f"Expected {n} samples, got {len(representative_sample)}"
    return representative_sample

# Example usage
n_samples = 50
representative_sample = draw_representative_sample(data, labels, n_samples)
print("Representative sample:")
representative_sample.head()


# %%
def get_random_sample(data: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    return data.sample(n=sample_size, random_state=1)

# Assuming 'data' is the full dataset and 'representative_sample' is already defined
random_sample = get_random_sample(data, len(representative_sample))

print("Random sample:")
random_sample.head()
# %%

# compare subset with full dataset. 
# Calculate the mean of all numerical columns for the full dataset
full_dataset_mean = data_wo_id.mean(numeric_only=True)

# Calculate the mean of all numerical columns for the representative sample
representative_sample_mean = representative_sample.mean(numeric_only=True)

random_sample_mean = random_sample.mean(numeric_only=True)

# Combine the means into a single DataFrame
comparison_df = pd.DataFrame({
    "Full Dataset Mean": full_dataset_mean,
    "Representative Sample Mean": representative_sample_mean,
    "Random Sample Mean": random_sample_mean
})

print("Comparison of means:")
comparison_df

# %%
# Show the distribution of difficulties for each dataframe but normalized

# Calculate the distribution of difficulties for the full dataset
full_dataset_difficulty_distribution = data['difficulty'].value_counts(normalize=True)

# Calculate the distribution of difficulties for the representative sample
representative_sample_difficulty_distribution = representative_sample['difficulty'].value_counts(normalize=True)

random_sample_difficulty_distribution = random_sample['difficulty'].value_counts(normalize=True)

print(full_dataset_difficulty_distribution)
print(representative_sample_difficulty_distribution)
print(random_sample_difficulty_distribution)


# %%

# compare the average correlation across both dataframes

corr = np.corrcoef(data_raw)
print(corr.shape)
print(corr)
print(np.nanmean(corr))

# %%

plt.imshow(corr)
plt.colorbar()
plt.show()
# %%
representative_sample.head()
representative_sample_raw = representative_sample.drop(columns=["id"]).fillna(0)

corr_rs = np.corrcoef(representative_sample_raw)
print(np.nanmean(corr_rs))

# %%

plt.imshow(corr_rs)
plt.colorbar()
plt.show()

# %%

random_sample_raw = random_sample.drop(columns=["id"]).fillna(0)

plt.imshow(np.corrcoef(random_sample_raw))
plt.colorbar()
plt.show()

# %%

def compare_correlation_patterns(full_data, subset_data):
    # Calculate correlation matrices
    full_corr = np.corrcoef(full_data.T)
    subset_corr = np.corrcoef(subset_data.T)
    
    # Get upper triangular values (excluding diagonal)
    full_corr_vals = full_corr[np.triu_indices(full_corr.shape[0], k=1)]
    subset_corr_vals = subset_corr[np.triu_indices(subset_corr.shape[0], k=1)]
    
    # Compare distributions of correlation values
    return full_corr_vals.mean(), subset_corr_vals.mean()

print(compare_correlation_patterns(data_raw, representative_sample_raw))
print(compare_correlation_patterns(data_raw, random_sample_raw))

# %%
# chat with claude about better metrics
# compare all of this to a random sample of the same size


# %%

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulate data similar to yours (21 dimensions)
np.random.seed(42)

# Standardize both datasets
scaler = StandardScaler()
full_scaled = scaler.fit_transform(data_wo_id.fillna(0))
subset_scaled = scaler.transform(representative_sample_raw.fillna(0))  # Use same scaling as full data
random_scaled = scaler.transform(random_sample_raw.fillna(0))
# Fit PCA on both datasets
pca_full = PCA(n_components=5)
pca_subset = PCA(n_components=5)
pca_random = PCA(n_components=5)

full_transformed = pca_full.fit_transform(full_scaled)
subset_transformed = pca_subset.fit_transform(subset_scaled)
random_transformed = pca_random.fit_transform(random_scaled)

# Compare explained variance ratios
print("Explained variance ratios:")
print("\nFull dataset:")
print([f"{var:.3f}" for var in pca_full.explained_variance_ratio_[:5]])
print("\nSubset:")
print([f"{var:.3f}" for var in pca_subset.explained_variance_ratio_[:5]])
print("\nRandom:")
print([f"{var:.3f}" for var in pca_random.explained_variance_ratio_[:5]])

# Compare component directions (absolute cosine similarity)
def cosine_similarity(v1, v2):
    return np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

print("\nCosine similarity between principal components (representative):")
for i in range(5):  # Compare first 5 components
    sim = cosine_similarity(pca_full.components_[i], pca_subset.components_[i])
    print(f"PC{i+1}: {sim:.3f}")

print("\nCosine similarity between principal components (random):")
for i in range(5):  # Compare first 5 components
    sim = cosine_similarity(pca_full.components_[i], pca_random.components_[i])
    print(f"PC{i+1}: {sim:.3f}")

# Calculate cumulative explained variance
cum_var_full = np.cumsum(pca_full.explained_variance_ratio_)
cum_var_subset = np.cumsum(pca_subset.explained_variance_ratio_)
cum_var_random = np.cumsum(pca_random.explained_variance_ratio_)

print("\nCumulative explained variance at 5 components:")
print(f"Full: {cum_var_full[4]:.3f}")
print(f"Subset: {cum_var_subset[4]:.3f}")
print(f"Random: {cum_var_random[4]:.3f}")
# %%

representative_sample.to_csv("data/representative_sample.csv")
# %%
# just ids
representative_sample_ids = representative_sample["id"]

import json

with open('data/representative_sample_ids.json', 'w') as f:
    json.dump(representative_sample_ids.tolist(), f)
# %%
