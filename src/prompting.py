def build_gsm8k_prompt(question: str) -> str:
    """
    Builds a consistent prompt for GSM8K.
    We format it without the final answer for generation/inference.
    """
    # For Gemma 4 E4B, we assume standard user/assistant structure if using chat template.
    # We will format this into a dictionary format and let the tokenizer's chat template handle it,
    # OR we can manually construct it if we want strict control over special tokens.
    # Since we need exact control over loss, let's use the standard Gemma text format.
    # Gemma uses <start_of_turn>user\n ... <end_of_turn>\n<start_of_turn>model\n ... <end_of_turn>
    return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

def extract_final_answer(generated_text: str) -> str:
    """
    Robustly extracts the final answer from GSM8K generations.
    GSM8K gold answers end with `#### [number]`.
    Generations might have `#### [number]` or just a number at the end.
    """
    if "####" in generated_text:
        return generated_text.split("####")[-1].strip()
    
    # Fallback heuristic: find the last number in the text
    import re
    numbers = re.findall(r'-?\d+\.?\d*', generated_text)
    if numbers:
        return numbers[-1]
    
    return ""
