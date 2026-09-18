import gradio as gr
import torch
import os
import sys

# Assume models are loaded inside the function to save initial startup time in demo
def solve_question(question, method, rank):
    # This is a stub for the demo UI.
    # In a real deployed app, models would be pre-loaded or loaded dynamically
    return "This is a placeholder for reasoning...", "42"

with gr.Blocks(title="LoRA vs DoRA on Gemma 4 E4B") as demo:
    gr.Markdown("# LoRA vs DoRA on Gemma 4 E4B - GSM8K")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="GSM8K Question", lines=4)
            method_selector = gr.Radio(choices=["LoRA", "DoRA"], value="LoRA", label="Method")
            rank_selector = gr.Radio(choices=["4", "8", "16", "32"], value="8", label="Rank")
            solve_btn = gr.Button("Solve", variant="primary")
            
        with gr.Column():
            reasoning_output = gr.Textbox(label="Reasoning", lines=6, interactive=False)
            answer_output = gr.Textbox(label="Extracted Final Answer", interactive=False)
            
    solve_btn.click(
        fn=solve_question,
        inputs=[question_input, method_selector, rank_selector],
        outputs=[reasoning_output, answer_output]
    )

if __name__ == "__main__":
    demo.launch()
