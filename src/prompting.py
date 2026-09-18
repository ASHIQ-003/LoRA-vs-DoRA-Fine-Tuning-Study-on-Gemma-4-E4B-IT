import re

def build_gsm8k_prompt(question: str) -> str:
    """
    Builds a canonical prompt for GSM8K.
    Used for baseline, LoRA, DoRA, and evaluation consistently.
    """
    return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

def normalize_number(num_str: str) -> str:
    """
    Normalizes a number string by removing commas and handling decimal points.
    """
    num_str = num_str.replace(",", "")
    if "." in num_str:
        # Strip trailing zeros and decimal if it becomes integer
        num_str = num_str.rstrip("0").rstrip(".")
    return num_str

def extract_final_answer(generated_text: str) -> str:
    """
    Robustly extracts the final answer from GSM8K generations.
    """
    if "####" in generated_text:
        ans = generated_text.split("####")[-1].strip()
        return normalize_number(ans)
    
    # Fallback heuristic: find the last number in the text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', generated_text)
    if numbers:
        return normalize_number(numbers[-1])
    
    return ""
