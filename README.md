# make_swe_bench_verified_mini

SWEBench-verified[TODO](TODO) is a great dataset, but it is very large. Thus, I wanted to create a smaller version that is a good proxy for the overall performance of models on the dataset. \

SWEBench-verified-mini is a subset of SWEBench-verified that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB of storage and has approximately the same distribution of performance, test pass rates and difficulty as the original dataset.

Note that I'm primarily combining work from other people, so I think most of the credit should go to others.
- TODO for creating the original dataset.
- TODO for taking the dataset and creating SWEBench-verified. 
- My MATS scholar Govind and Axel for running the full dataset on a decent number of models for a different project.
- TODO for figuring out how to build tiny versions of bigger benchmarks (I use a different approach than them but was inspired by their work).

## How to make SWEBench-verified-mini

We want to select 50 datapoints of the 500 datapoints in SWEBench-verified such that:
1. The marginal distribution of important scores (performance, test pass rates, difficulty) are similar to the full dataset. 
2. The overall storage sizes of the dataset is minimized. SWEBench-verified requires 130GB or 260GB of storage (depending on whether you want to create ARM and AMD versions of the dataset). TODO double check this. This is because you need to create 40 different environments. We want to find a subset of dataset that is a good proxy while using as few environments as possible.

## How to run the code

If you want to run the code, you can either run the file run_all.py or run the individual files. In all cases, you have to add the .eval files of previous runs on the full SWEBench-verified dataset to the /logs folder.

If you want to run the individual files, you have to start with get_docker_image_sizes.py and extract_data_from_logs.py. These are data wrangling scripts.

Then you can run add_metadata_to_data.py and generate_subsets.py.

If you want to compare the generated subsets, you can run compare_subsets.py.

Finally, you can run make_new_huggingface_dataset.py to create the new dataset.

