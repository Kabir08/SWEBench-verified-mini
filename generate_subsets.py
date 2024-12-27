from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import resample
import pulp

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data by mapping difficulty and creating numeric version.
    
    Parameters:
    - data: Raw DataFrame with original columns
    
    Returns:
    - Processed numeric DataFrame
    """
    # Map difficulty to numeric values
    difficulty_map = {
        '<15 min fix': 0,
        '15 min - 1 hour': 1,
        '1-4 hours': 2,
        '>4 hours': 3
    }
    data = data.copy()
    data["difficulty"] = data["difficulty"].map(difficulty_map)
    
    # Create numeric version for clustering
    data_numeric = data.drop(columns=["id", "size_in_gb", "environment"])
    data_numeric = data_numeric.fillna(0)
    
    return data_numeric

def compute_kmeans(data: pd.DataFrame, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute k-means clustering on the given data.
    
    Parameters:
    - data: DataFrame containing the data to cluster
    - n_clusters: Number of clusters to form
    
    Returns:
    - Cluster labels and centers
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def draw_representative_sample(data: pd.DataFrame, labels: np.ndarray, n: int, random_state: int = 1) -> pd.DataFrame:
    """
    Draw N representative datapoints based on k-means clustering labels.
    First selects the point closest to each cluster centroid,
    then distributes remaining samples proportionally.
    
    Parameters:
    - data: DataFrame containing the data to sample from
    - labels: Cluster labels for each data point
    - n: Number of representative datapoints to draw
    - random_state: Random seed for reproducibility
    
    Returns:
    - DataFrame containing the representative sample
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # First, get the point closest to centroid for each cluster
    data_numeric = prepare_data(data)
    centroid_samples = pd.DataFrame()
    remaining_data = data.copy()
    remaining_indices = np.ones(len(data), dtype=bool)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data_numeric)
    centroids = kmeans.cluster_centers_
    
    # Select points closest to centroids
    for label, centroid in zip(unique_labels, centroids):
        cluster_mask = labels == label
        cluster_data_numeric = data_numeric[cluster_mask]
        
        # Calculate distances to centroid
        distances = np.linalg.norm(cluster_data_numeric - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get the original index in the full dataset
        original_idx = np.where(cluster_mask)[0][closest_idx]
        
        # Add to centroid samples and mark for removal from remaining data
        centroid_samples = pd.concat([centroid_samples, data.iloc[[original_idx]]])
        remaining_indices[original_idx] = False
    
    # Update remaining data
    remaining_data = data[remaining_indices]
    remaining_labels = labels[remaining_indices]
    
    # Calculate proportional distribution for remaining samples
    remaining_n = n - n_clusters
    if remaining_n > 0:
        # Count remaining points per cluster
        unique_remaining_labels, counts = np.unique(remaining_labels, return_counts=True)
        proportions = counts / counts.sum()
        
        # Calculate samples per cluster
        exact_samples = proportions * remaining_n
        sample_counts = exact_samples.astype(int)
        fractions = exact_samples - sample_counts
        
        # Distribute remaining samples based on largest fractional parts
        remaining = remaining_n - sample_counts.sum()
        if remaining > 0:
            fraction_indices = np.argsort(fractions)[-remaining:]
            for idx in fraction_indices:
                sample_counts[idx] += 1
        
        # Sample from remaining points
        additional_samples = pd.DataFrame()
        for label, count in zip(unique_remaining_labels, sample_counts):
            if count > 0:
                cluster_data = remaining_data[remaining_labels == label]
                count = min(count, len(cluster_data))
                if count > 0:
                    cluster_sample = resample(cluster_data, 
                                           n_samples=count, 
                                           random_state=random_state, 
                                           replace=False)
                    additional_samples = pd.concat([additional_samples, cluster_sample])
        
        # Combine centroid samples with additional samples
        representative_sample = pd.concat([centroid_samples, additional_samples])
    else:
        representative_sample = centroid_samples
    
    return representative_sample

def optimize_cluster_selection(data: pd.DataFrame, labels: np.ndarray, n_samples: int = 50) -> pd.DataFrame:
    """
    Select representative samples while minimizing environment size footprint.
    
    Parameters:
    - data: DataFrame containing the full dataset
    - labels: Cluster labels from k-means
    - n_samples: Number of samples to select
    
    Returns:
    - DataFrame containing the optimally selected samples
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    
    # Calculate samples per cluster
    exact_samples = proportions * n_samples
    sample_counts = exact_samples.astype(int)
    fractions = exact_samples - sample_counts
    
    # Distribute remaining samples
    remaining = n_samples - sample_counts.sum()
    if remaining > 0:
        fraction_indices = np.argsort(fractions)[-remaining:]
        for idx in fraction_indices:
            sample_counts[idx] += 1
            
    # Create optimization problem
    prob = pulp.LpProblem("Minimize_Environment_Size", pulp.LpMinimize)
    
    # Create mapping of datapoint indices to environments and sizes
    env_mapping = {i: env for i, env in enumerate(data['environment'])}
    unique_envs = data['environment'].unique()
    env_sizes = {env: size for env, size in 
                zip(data['environment'], data['size_in_gb'])}
    
    # Decision variables
    x = pulp.LpVariable.dicts("select",
                        ((i, j) for i in unique_labels 
                         for j in np.where(labels == i)[0]),
                        cat='Binary')
    
    y = pulp.LpVariable.dicts("env",
                        unique_envs,
                        cat='Binary')
    
    # Objective: minimize total environment size
    prob += pulp.lpSum(env_sizes[k] * y[k] for k in unique_envs)
    
    # Constraint: must select correct number from each cluster
    for i, target in zip(unique_labels, sample_counts):
        prob += pulp.lpSum(x[i,j] for j in np.where(labels == i)[0]) == target
    
    # Constraint: if point selected, its environment must be selected
    for i in unique_labels:
        for j in np.where(labels == i)[0]:
            env = env_mapping[j]
            prob += x[i,j] <= y[env]
    
    # Solve
    status = prob.solve()
    
    if status != 1:
        raise ValueError("No optimal solution found")
    
    # Extract selected points
    selected_indices = []
    for i in unique_labels:
        for j in np.where(labels == i)[0]:
            if pulp.value(x[i,j]) > 0.5:
                selected_indices.append(j)
    
    selected_envs = [env for env in unique_envs if pulp.value(y[env]) > 0.5]
    total_size = sum(env_sizes[env] for env in selected_envs)
    print(f"Total environment size: {total_size:.2f} GB")
    print(f"Number of environments: {len(selected_envs)}")
    
    return data.iloc[selected_indices].copy()

def get_random_sample(data: pd.DataFrame, sample_size: int, random_state: int = 1) -> pd.DataFrame:
    """
    Get a random sample from the dataset.
    
    Parameters:
    - data: DataFrame to sample from
    - sample_size: Number of samples to draw
    
    Returns:
    - Random sample DataFrame
    """
    return data.sample(n=sample_size, random_state=random_state)

def save_sample_and_ids(sample: pd.DataFrame, name: str, data_dir: Path) -> None:
    """
    Save a sample DataFrame and its IDs to files.
    
    Parameters:
    - sample: DataFrame to save
    - name: Name prefix for the files
    - data_dir: Directory to save the files
    """
    # Save full sample
    sample.to_csv(data_dir / f"{name}.csv", index=False)
    
    # Save IDs
    with open(data_dir / f"{name}_ids.json", 'w') as f:
        json.dump(sample["id"].tolist(), f)

def main() -> None:
    """Generate and save different samples from the dataset."""
    # Load data
    data = pd.read_csv(Path("data") / "merged_df_pass_rate_with_metadata.csv")
    data_numeric = prepare_data(data)
    
    # Generate clusters
    n_clusters = 10
    labels, _ = compute_kmeans(data_numeric, n_clusters)
    #print("Label counts:", np.bincount(labels))
    
    # Generate samples
    n_samples = 50
    representative_sample = draw_representative_sample(data, labels, n_samples, random_state=1)
    size_optimized_sample = optimize_cluster_selection(data, labels, n_samples)
    random_sample = get_random_sample(data, n_samples, random_state=1)
    
    # Save samples
    data_dir = Path("data/subsets")
    save_sample_and_ids(representative_sample, "representative_sample", data_dir)
    save_sample_and_ids(size_optimized_sample, "size_optimized_sample", data_dir)
    save_sample_and_ids(random_sample, "random_sample", data_dir)

if __name__ == "__main__":
    main()