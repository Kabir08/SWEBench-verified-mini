# %%
from pathlib import Path
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

eval_file_names = [
        "claude_opus3.eval",
        "claude_sonnet3.eval",
        "claude_sonnet35.eval",
        "claude_sonnet36.eval",
        "gpt4o.eval",
        "gpt4o-mini.eval",
        "gpt4turbo.eval",
    ]


logs = [list_eval_logs(str(Path("logs") / eval_file_name)) for eval_file_name in eval_file_names]
eval_logs = [read_eval_log(x[0]) for x in logs]

# %%
sample = eval_logs[0].samples[0]
explanation = sample.scores["swe_bench_scorer"].explanation
print(explanation)

# %%

def count_test_transitions(test_output: str) -> tuple[tuple[int, int], tuple[int, int]]:
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
    pass_to_pass_passed = pass_section.count('"PASSED"')
    pass_to_pass_failed = pass_section.count('"FAILED"')

    
    fail_to_pass_passed = fail_section.count('"PASSED"')
    fail_to_pass_failed = fail_section.count('"FAILED"')
    
    return ((pass_to_pass_passed, pass_to_pass_failed), 
            (fail_to_pass_passed, fail_to_pass_failed))

# Example usage:
pass_to_pass, fail_to_pass = count_test_transitions(explanation)
print(f"PASS_TO_PASS: {pass_to_pass[0]} passed, {pass_to_pass[1]} failed")
print(f"FAIL_TO_PASS: {fail_to_pass[0]} passed, {fail_to_pass[1]} failed")

# %%
