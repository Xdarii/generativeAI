import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def generate_response(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=20)

    # Decode and return the response
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded_output

# Interface for Gradio
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    live=False,
    title="FLAN-T5 Text Generation",
    description="Enter a prompt, and FLAN-T5 will generate a response.",
)

# Launch Gradio interface
iface.launch()
