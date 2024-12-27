from pathlib import Path

# Import functions from each module
from get_docker_image_sizes import main as get_docker_sizes
from extract_data_from_logs import main as extract_data
from add_metadata_to_data import main as add_metadata
from generate_subsets import main as generate_subsets
from compare_subsets import main as compare_subsets
from make_new_huggingface_dataset import main as make_dataset

def check_file_exists(file_path: Path) -> bool:
    """Check if a file exists and print status."""
    exists = file_path.exists()
    print(f"Checking {file_path}: {'✓' if exists else '✗'}")
    return exists

def verify_prerequisites() -> tuple[bool, str]:
    """
    Verify all required input files exist.
    
    Returns:
    - Tuple of (success status, error message if any)
    """
    required_files = [
        Path("data/external_data/docker_terminal_output.csv"),
        Path("data/external_data/instance_to_env.json"),
        Path("data/external_data/ensembled_annotations_public.csv"),
        Path("logs")  # Directory containing evaluation logs
    ]
    
    print("\nVerifying prerequisites...")
    all_exist = all(check_file_exists(f) for f in required_files)
    
    if not all_exist:
        return False, "Missing required input files"
    return True, ""

def create_directories() -> None:
    """Create necessary output directories if they don't exist."""
    directories = [
        Path("data"),
        Path("data/subsets"),
        Path("data/filtered_huggingface_dataset")
    ]
    
    print("\nCreating output directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created {directory}")

def main() -> None:
    """
    Run all steps to create SWE-bench-verified-mini dataset.
    
    Steps:
    1. Verify prerequisites
    2. Get Docker image sizes
    3. Extract data from logs
    4. Add metadata to data
    5. Generate subsets
    6. Compare subsets
    7. Create new HuggingFace dataset
    """
    print("Starting SWE-bench-verified-mini creation process...")
    
    # Check prerequisites
    success, error = verify_prerequisites()
    if not success:
        print(f"Error: {error}")
        return
    
    # Create necessary directories
    create_directories()
    
    # Run each step
    steps = [
        ("Getting Docker image sizes", get_docker_sizes),
        ("Extracting data from logs", extract_data),
        ("Adding metadata to data", add_metadata),
        ("Generating subsets", generate_subsets),
        ("Comparing subsets", compare_subsets),
        ("Creating HuggingFace dataset", make_dataset)
    ]
    
    for step_name, step_function in steps:
        print(f"\n{'='*50}")
        print(f"Step: {step_name}")
        print('='*50)
        try:
            step_function()
            print(f"\n✓ Completed: {step_name}")
        except Exception as e:
            print(f"\n✗ Error in {step_name}: {str(e)}")
            print("Stopping execution.")
            return
    
    print("\n✓ All steps completed successfully!")
    print("\nSWE-bench-verified-mini dataset has been created and pushed to HuggingFace.")

if __name__ == "__main__":
    main()
