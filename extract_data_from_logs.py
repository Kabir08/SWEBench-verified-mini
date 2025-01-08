# %%
from pathlib import Path
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
import numpy as np
def count_test_results(test_output: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Count passed and failed tests in both PASS_TO_PASS and FAIL_TO_PASS sections.
    
    Args:
        test_output: String containing test results
        
    Returns:
        tuple of tuples: ((pass_to_pass_passed, pass_to_pass_failed), 
                         (fail_to_pass_passed, fail_to_pass_failed))
    """
    # Split at FAIL_TO_PASS
    pass_section, fail_section = test_output.split("FAIL_TO_PASS:")
    
    # Count results in each section
    pass_to_pass_passed = np.float64(pass_section.count('"PASSED"'))
    pass_to_pass_failed = np.float64(pass_section.count('"FAILED"'))

    fail_to_pass_passed = np.float64(fail_section.count('"PASSED"'))
    fail_to_pass_failed = np.float64(fail_section.count('"FAILED"'))
    
    return ((pass_to_pass_passed, pass_to_pass_failed), 
            (fail_to_pass_passed, fail_to_pass_failed))

def get_score_and_test_counts_from_sample(sample) -> tuple[float, float, float, float, float]:
    """Extract score and test counts, handling invalid UTF-8 data."""
    # Get score first since it doesn't depend on the explanation string

    try:
        score = sample.scores["swe_bench_scorer"].value
    except Exception as e:
        print(f"Error getting score for sample {sample.id}: {str(e)}")
        score = float('nan')
    
    try:
        explanation = sample.scores["swe_bench_scorer"].explanation
        if not isinstance(explanation, str):
            print(f"Warning: Invalid explanation format for sample {sample.id}")
            return score, float('nan'), float('nan'), float('nan'), float('nan')
            
        (pass_to_pass_passed, pass_to_pass_failed), (fail_to_pass_passed, fail_to_pass_failed) = count_test_results(explanation)
        
    except UnicodeDecodeError:
        print(f"Warning: Invalid UTF-8 data in sample {sample.id}")
        return score, float('nan'), float('nan'), float('nan'), float('nan')
    except Exception as e:
        print(f"Error processing sample {sample.id}: {str(e)}")
        return score, float('nan'), float('nan'), float('nan'), float('nan')
        
    return score, pass_to_pass_passed, pass_to_pass_failed, fail_to_pass_passed, fail_to_pass_failed

def merge_model_scores(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    transformed_dfs = []
    
    for df in df_list:
        model_name = df['model'].iloc[0]
        transformed_df = df[['id', 'score']].copy()
        transformed_df = transformed_df.rename(columns={'score': model_name})
        transformed_df = transformed_df.sort_values('id')
        transformed_dfs.append(transformed_df)
    
    result = transformed_dfs[0]
    for df in transformed_dfs[1:]:
        result = result.merge(df, on='id', how='outer')
    
    return result.sort_values('id')

def merge_model_pass_rates(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    transformed_dfs = []
    
    for df in df_list:
        model_name = df['model'].iloc[0]
        pass_to_pass_total = df['pass_to_pass_passed'] + df['pass_to_pass_failed']
        fail_to_pass_total = df['fail_to_pass_passed'] + df['fail_to_pass_failed']
        pass_to_pass_rate = df['pass_to_pass_passed'] / pass_to_pass_total
        fail_to_pass_rate = df['fail_to_pass_passed'] / fail_to_pass_total
        transformed_df = pd.DataFrame({
            'id': df['id'],
            f'{model_name}_score': df['score'],
            f'{model_name}_pass_to_pass_pass_rate': pass_to_pass_rate,
            f'{model_name}_fail_to_pass_pass_rate': fail_to_pass_rate
        })
            
        transformed_dfs.append(transformed_df)
    
    result = transformed_dfs[0]
    for df in transformed_dfs[1:]:
        result = result.merge(df, on='id', how='outer')
    
    return result.sort_values('id')



def main():
    eval_file_names = [
        "Qwen2.5-32B-Coder.eval",
        "Qwen2.5-7B.eval",
        "claude_opus3.eval",
        "claude_sonnet35.eval",
        "gpt35turbo.eval",
        "gpt4o.eval",
        "llama3.1-405b.eval",
        "llama3.1-8b.eval",
        "Qwen2.5-72B.eval",
        "claude21.eval",
        "claude_sonnet3.eval",
        "claude_sonnet36.eval",
        "gpt4o-mini.eval",
        "gpt4turbo.eval",
        "llama3.1-70b.eval",
        "o1.eval",
    ]

    logs = [list_eval_logs(str(Path("logs") / eval_file_name)) for eval_file_name in eval_file_names]
    eval_logs = [read_eval_log(x[0]) for x in logs]

    log_dfs = []
    for log in eval_logs:
        model_name = log.eval.model
        print(f"Processing {model_name}...")
        values_list = []
        for sample in log.samples:
            score, pass_to_pass_passed, pass_to_pass_failed, fail_to_pass_passed, fail_to_pass_failed = get_score_and_test_counts_from_sample(sample)
            values = {"id": sample.id, "model": model_name, "score": score, "pass_to_pass_passed": pass_to_pass_passed, "pass_to_pass_failed": pass_to_pass_failed, "fail_to_pass_passed": fail_to_pass_passed, "fail_to_pass_failed": fail_to_pass_failed}
            
            values_list.append(values)
        
        values_df = pd.DataFrame(values_list)
        log_dfs.append(values_df)

    merged_df = merge_model_scores(log_dfs)
    print(merged_df)
    merged_df.to_csv("data/merged_df.csv", index=False)

    merged_df_pass_rate = merge_model_pass_rates(log_dfs)
    print(merged_df_pass_rate)
    merged_df_pass_rate.to_csv("data/merged_df_pass_rate.csv", index=False)
    print(merged_df_pass_rate.head())


if __name__ == "__main__":
    main()


# %%
