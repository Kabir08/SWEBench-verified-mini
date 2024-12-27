from pathlib import Path
import json
import pandas as pd

def convert_to_gb(size_str: str) -> float:
    """
    Convert a size string (GB, MB, or kB) to GB float value.
    
    Parameters:
    - size_str: String containing size with unit (e.g., "77.9MB", "2.58GB", "500kB")
        
    Returns:
    - Size in GB as float
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

def process_docker_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the raw Docker output DataFrame to standardize column names and formats.
    
    Parameters:
    - df: Raw DataFrame from Docker output
    
    Returns:
    - Processed DataFrame with standardized columns
    """
    # Map columns to their new names
    df["IMAGE ID"] = df["IMAGE"].astype(str)
    df["CREATED"] = df["ID"].astype(str) + " " + df["CREATED"].astype(str) + " " + df["SIZE"].astype(str)
    df["SIZE"] = df["SHARED"].astype(str)
    df["SHARED SIZE"] = df["SIZE.1"].astype(str)
    df["UNIQUE SIZE"] = df["UNIQUE"].astype(str)
    df["CONTAINERS"] = df["SIZE.2"].astype(str)

    # Drop redundant columns
    df = df.drop(columns=['IMAGE', 'ID', 'SHARED', 'SIZE.1', 'UNIQUE', 'SIZE.2'])

    # Reorder columns
    correct_order = ["REPOSITORY", "TAG", "IMAGE ID", "CREATED", "SIZE", 
                    "SHARED SIZE", "UNIQUE SIZE", "CONTAINERS"]
    df = df[correct_order]

    # Extract ID from repository name
    df["ID"] = df["REPOSITORY"].str.replace("sweb.eval.x86_64.", "", regex=False)

    # Add size in GB column
    df["SIZE_IN_GB"] = df["SIZE"].apply(convert_to_gb)

    return df

def map_environments(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """
    Map Docker image IDs to their environments using a JSON mapping file.
    
    Parameters:
    - df: DataFrame containing Docker image information
    - mapping_path: Path to the JSON file containing instance to environment mapping
    
    Returns:
    - DataFrame with added environment mapping
    """
    with mapping_path.open("r") as file:
        instance_to_env = json.load(file)

    df["ENVIRONMENT"] = df["ID"].map(instance_to_env)
    return df

def main() -> None:
    """Process Docker image data and save results."""
    external_data_dir = Path("data/external_data")
    
    # Read and process Docker output
    df = pd.read_csv(external_data_dir / "docker_terminal_output.csv")
    df = process_docker_output(df)
    
    # Map environments
    df = map_environments(df, external_data_dir / "instance_to_env.json")
    
    # Reset index and save
    df = df.reset_index(drop=True)
    df.to_csv(external_data_dir / "docker_image_sizes.csv", index=False)
    
    # Print summary
    print("\nProcessed DataFrame:")
    print(df.head())
    print(f"\nNumber of unique environments: {df['ENVIRONMENT'].nunique()}")

if __name__ == "__main__":
    main()



# %%
