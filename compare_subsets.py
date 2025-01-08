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
                 random: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    """
    Compare means of different samples against full dataset using subplots.
    Metrics are grouped by type (scores, fail_to_pass, pass_to_pass).
    """
    # Calculate means
    means_df = pd.DataFrame({
        "Full Dataset": full_data.mean(numeric_only=True),
        "K-means Representative": representative.mean(numeric_only=True),
        "Random": random.mean(numeric_only=True),
        "Size Optimized": optimized.mean(numeric_only=True)
    })
    
    # Group metrics by type
    score_metrics = [col for col in means_df.index if col.endswith('_score')]
    fail_to_pass_metrics = [col for col in means_df.index if col.endswith('_fail_to_pass_pass_rate')]
    pass_to_pass_metrics = [col for col in means_df.index if col.endswith('_pass_to_pass_pass_rate')]
    
    def plot_group(ax, metrics, title, show_labels=False):
        bar_width = 0.2
        r = np.arange(len(metrics))
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)
        
        for idx, (sample_name, means) in enumerate(means_df.items()):
            position = r + idx * bar_width
            ax.bar(position, means[metrics], bar_width, label=sample_name, zorder=3)
        
        ax.set_ylabel('Mean Value')
        ax.set_title(title)
        ax.set_xticks(r + bar_width * (len(means_df.columns)-1) / 2)
        
        if show_labels:
            # For the last row, only show model names
            labels = [m.split('_')[0] for m in metrics]
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])  # Empty labels for first two rows
        
        # Only show legend for the top subplot
        if ax == ax1:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Create figure with 3 subplots sharing x axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8), sharex=False)
    
    # Plot each group
    plot_group(ax1, score_metrics, 'Model Scores', show_labels=False)
    plot_group(ax2, fail_to_pass_metrics, 'Fail to Pass Rates', show_labels=False)
    plot_group(ax3, pass_to_pass_metrics, 'Pass to Pass Rates', show_labels=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(Path('figures') / 'means_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return means_df

def compare_difficulty_distributions(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare and plot difficulty distributions across samples."""
    distributions = {}
    for name, df in samples.items():
        dist = df['difficulty'].value_counts(normalize=True)
        distributions[name] = dist
    
    # Create a DataFrame from the distributions dictionary
    dist_df = pd.DataFrame(distributions).fillna(0)
    
    # Plot the distributions
    plt.figure(figsize=(12, 4))
    ax = dist_df.plot(kind='bar')
    
    # Add grid and ensure it's behind the bars
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # Update the zorder of bars by accessing the patches directly
    for patch in ax.patches:
        patch.set_zorder(3)
    
    plt.title('Difficulty Distributions Across Samples')
    plt.xlabel('Difficulty')
    plt.ylabel('Proportion')
    plt.legend(title='Sample')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(Path('figures') / 'difficulty_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
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