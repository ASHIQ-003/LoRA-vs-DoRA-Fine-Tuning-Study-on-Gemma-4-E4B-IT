from src.prompting import extract_final_answer, build_gsm8k_prompt

def test_extract_final_answer():
    # Test strict format
    text1 = "The answer is 42. #### 42"
    assert extract_final_answer(text1) == "42"
    
    # Test integer normalization
    text2 = "Some text #### 42.0"
    assert extract_final_answer(text2) == "42"
    
    # Test comma normalization
    text3 = "Text #### 1,000"
    assert extract_final_answer(text3) == "1000"
    
    # Test decimal normalization
    text4 = "Text #### 1,000.50"
    assert extract_final_answer(text4) == "1000.5"
    
    # Test malformed fallback (no ####)
    text5 = "I think the answer is 1,234.50"
    assert extract_final_answer(text5) == "1234.5"
    
    # Negative number
    text6 = "It drops by #### -5"
    assert extract_final_answer(text6) == "-5"

def test_build_gsm8k_prompt():
    prompt = build_gsm8k_prompt("What is 2+2?")
    assert prompt == "<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"

if __name__ == "__main__":
    test_extract_final_answer()
    test_build_gsm8k_prompt()
    print("All evaluator tests passed.")
