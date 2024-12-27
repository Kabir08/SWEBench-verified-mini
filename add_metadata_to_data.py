from pathlib import Path
import pandas as pd

def merge_data_with_metadata(data: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """
    Merge data with metadata annotations based on ID.
    
    Parameters:
    - data: DataFrame containing model performance data
    - annotation: DataFrame containing metadata annotations
    
    Returns:
    - DataFrame with merged data and metadata
    """
    # Get IDs from data
    ids = data["id"]
    
    # Filter metadata to only include relevant IDs
    subset_meta_data = annotation[annotation["instance_id"].isin(ids)]
    
    # Merge data with metadata
    data_with_metadata = data.copy()
    data_with_metadata["difficulty"] = data_with_metadata["id"].map(
        subset_meta_data.set_index("instance_id")["difficulty"]
    )
    
    return data_with_metadata

def merge_with_environment_data(df: pd.DataFrame, size_env_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge data with environment size information.
    
    Parameters:
    - df: DataFrame containing model performance and difficulty data
    - size_env_data: DataFrame containing environment size information
    
    Returns:
    - DataFrame with added environment information
    """
    # Prepare environment data
    size_env_subset = size_env_data[["ID", "SIZE_IN_GB", "ENVIRONMENT"]]
    size_env_subset.columns = ["id", "size_in_gb", "environment"]
    
    # Merge with main data
    merged_data = df.merge(size_env_subset, on="id", how="left")
    return merged_data

def main() -> None:
    """Process and merge all metadata with the main dataset."""
    data_dir = Path("data")
    external_data_dir = data_dir / "external_data"
    
    # Load all required datasets
    data = pd.read_csv(data_dir / "merged_df.csv")
    annotation = pd.read_csv(external_data_dir / "ensembled_annotations_public.csv")
    pass_rate_data = pd.read_csv(data_dir / "merged_df_pass_rate.csv")
    size_env_data = pd.read_csv(external_data_dir / "docker_image_sizes.csv")
    
    # Print initial data info
    print("Annotation data:")
    print(annotation.head())
    print("\nEnvironment size data:")
    print(size_env_data.head())
    print(f"\nTotal annotations: {len(annotation)}")
    print("Annotation columns:", annotation.columns)
    
    # Process main data with metadata
    data_with_difficulties = merge_data_with_metadata(data, annotation)
    
    # Save intermediate result
    data_with_difficulties.to_csv(data_dir / "merged_df_with_difficulties.csv", index=False)
    print("\nData with difficulties:")
    print(data_with_difficulties.head())
    
    # Process pass rate data with metadata
    pass_rate_data_with_difficulties = merge_data_with_metadata(pass_rate_data, annotation)
    
    # Add environment data
    merged_data = merge_with_environment_data(
        pass_rate_data_with_difficulties, 
        size_env_data
    )
    
    # Print difficulty distribution
    difficulties = merged_data["difficulty"]
    print("\nDifficulty distribution:")
    print(difficulties.value_counts())
    print("\nDifficulty distribution (percentages):")
    print(difficulties.value_counts() / len(difficulties))
    
    # Save final result
    merged_data.to_csv(data_dir / "merged_df_pass_rate_with_metadata.csv", index=False)
    print("\nFinal merged data:")
    print(merged_data.head())

if __name__ == "__main__":
    main()