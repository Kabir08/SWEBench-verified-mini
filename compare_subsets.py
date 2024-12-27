from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepare_numeric_data(data: pd.DataFrame, replace_difficulty_with_number: bool = False) -> pd.DataFrame:
    """Prepare numeric data for analysis by removing non-numeric columns and filling NaNs."""
    if replace_difficulty_with_number:
        # Map difficulty to numeric values
        difficulty_map = {
            '<15 min fix': 0,
            '15 min - 1 hour': 1,
            '1-4 hours': 2,
            '>4 hours': 3
        }
        data = data.copy()
        data["difficulty"] = data["difficulty"].map(difficulty_map)
    
    return data.drop(columns=["id", "environment", "size_in_gb"]).fillna(0)

def compare_means(full_data: pd.DataFrame, representative: pd.DataFrame, 
                 random: pd.DataFrame, optimized: pd.DataFrame) -> None:
    """
    Compare means of different samples against full dataset using a bar plot.
    Each metric gets its own group of bars for direct comparison.
    """
    # Calculate means
    means_df = pd.DataFrame({
        "Full Dataset": full_data.mean(numeric_only=True),
        "Representative": representative.mean(numeric_only=True),
        "Random": random.mean(numeric_only=True),
        "Optimized": optimized.mean(numeric_only=True)
    })
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Number of metrics and samples
    n_metrics = len(means_df.index)
    n_samples = len(means_df.columns)
    
    # Set width of bars and positions of bar groups
    bar_width = 0.2
    r = np.arange(n_metrics)
    
    # Create bars for each sample type
    for idx, (sample_name, means) in enumerate(means_df.items()):
        position = r + idx * bar_width
        ax.bar(position, means, bar_width, label=sample_name)
    
    # Customize the plot
    ax.set_ylabel('Mean Value')
    ax.set_title('Comparison of Means Across Samples')
    ax.set_xticks(r + bar_width * (n_samples-1) / 2)
    ax.set_xticklabels(means_df.index, rotation=45, ha='right')
    ax.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    return means_df

def compare_difficulty_distributions(samples: dict[str, pd.DataFrame]) -> None:
    """Compare and plot difficulty distributions across samples."""
    distributions = {}
    for name, df in samples.items():
        dist = df['difficulty'].value_counts(normalize=True)
        distributions[name] = dist
    
    # Create a DataFrame from the distributions dictionary
    dist_df = pd.DataFrame(distributions).fillna(0)
    
    # Plot the distributions
    dist_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Difficulty Distributions Across Samples')
    plt.xlabel('Difficulty')
    plt.ylabel('Proportion')
    plt.legend(title='Sample')
    plt.show()

    return dist_df

def compare_correlation_patterns(full_data: pd.DataFrame, subset_data: pd.DataFrame) -> tuple[float, float]:
    """Compare correlation patterns between full dataset and subset."""
    full_corr = np.corrcoef(full_data.T)
    subset_corr = np.corrcoef(subset_data.T)
    
    # Get upper triangular values (excluding diagonal)
    full_corr_vals = full_corr[np.triu_indices(full_corr.shape[0], k=1)]
    subset_corr_vals = subset_corr[np.triu_indices(subset_corr.shape[0], k=1)]
    
    return np.nanmean(full_corr_vals), np.nanmean(subset_corr_vals)

def compare_pca_components(full_data: pd.DataFrame, samples: dict[str, pd.DataFrame], 
                         n_components: int = 5) -> None:
    """Compare PCA components between full dataset and samples."""
    # Standardize data
    scaler = StandardScaler()
    full_scaled = scaler.fit_transform(full_data)
    
    # Fit PCA on full dataset
    pca_full = PCA(n_components=n_components)
    pca_full.fit_transform(full_scaled)
    
    print("\nExplained variance ratios:")
    print("\nFull dataset:")
    print([f"{var:.3f}" for var in pca_full.explained_variance_ratio_])
    
    # Compare with samples
    for name, sample_df in samples.items():
        sample_scaled = scaler.transform(sample_df)
        pca_sample = PCA(n_components=n_components)
        pca_sample.fit_transform(sample_scaled)
        
        print(f"\n{name}:")
        print([f"{var:.3f}" for var in pca_sample.explained_variance_ratio_])
        
        # Compare component directions
        print(f"\nCosine similarity between principal components ({name}):")
        for i in range(n_components):
            sim = np.abs(np.dot(pca_full.components_[i], pca_sample.components_[i]) / 
                        (np.linalg.norm(pca_full.components_[i]) * 
                         np.linalg.norm(pca_sample.components_[i])))
            print(f"PC{i+1}: {sim:.3f}")

def calculate_environment_sizes(samples: dict[str, pd.DataFrame]) -> None:
    """Calculate and compare total environment sizes for each sample."""
    for name, df in samples.items():
        unique_envs = df['environment'].unique()
        env_sizes = df.drop_duplicates(subset=['environment'])[['environment', 'size_in_gb']]
        total_size = env_sizes['size_in_gb'].sum()
        print(f"\nSize of {name}: {total_size:.2f} GB")
        print(f"Number of unique environments in {name}: {len(unique_envs)}")

def main() -> None:
    """Compare different samples using various metrics."""
    # Load datasets
    data_dir = Path("data")
    subsets_dir = data_dir / "subsets"
    
    full_data = pd.read_csv(data_dir / "merged_df_pass_rate_with_metadata.csv")
    representative = pd.read_csv(subsets_dir / "representative_sample.csv")
    optimized = pd.read_csv(subsets_dir / "size_optimized_sample.csv")
    random = pd.read_csv(subsets_dir / "random_sample.csv")
    
    # Prepare numeric versions
    full_numeric = prepare_numeric_data(full_data)
    representative_numeric = prepare_numeric_data(representative)
    optimized_numeric = prepare_numeric_data(optimized)
    random_numeric = prepare_numeric_data(random)
    
    # Compare means
    means_df = compare_means(full_numeric, representative_numeric, 
                 random_numeric, optimized_numeric)

    print(means_df)
    
    # Compare difficulty distributions
    samples = {
        "Full Dataset": full_data,
        "Representative Sample": representative,
        "Random Sample": random,
        "Size Optimized Sample": optimized
    }
    difficulty_distributions = compare_difficulty_distributions(samples)

    print(difficulty_distributions)
    
    # Compare correlation patterns
    print("\nCorrelation pattern comparison (mean correlation):")
    for name, sample in samples.items():
        if name != "Full Dataset":
            full_corr, sample_corr = compare_correlation_patterns(
                prepare_numeric_data(full_data, replace_difficulty_with_number=True), 
                prepare_numeric_data(sample, replace_difficulty_with_number=True))
            print(f"{name}: {full_corr:.3f} vs {sample_corr:.3f}")
    
    # Compare PCA components
    numeric_samples = {name: prepare_numeric_data(df, replace_difficulty_with_number=True) for name, df in samples.items()}
    compare_pca_components(prepare_numeric_data(full_data, replace_difficulty_with_number=True), 
                         {k: v for k, v in numeric_samples.items() if k != "Full Dataset"})
    
    # Calculate environment sizes
    calculate_environment_sizes(samples)

if __name__ == "__main__":
    main()