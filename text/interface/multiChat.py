
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline, set_seed

# Load models and tokenizers
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

gpt2_generator = pipeline('text-generation', model='gpt2')

def generate_response(input_text, use_flan):
    if use_flan:
        model = flan_model
        tokenizer = flan_tokenizer
    else:
        model = gpt2_generator.model
        tokenizer = gpt2_generator.tokenizer

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
    inputs=[gr.Textbox(), gr.Checkbox(value="Use FLAN-T5")],
    outputs="text",
    live=False,
    title="Text Generation",
    description="Enter a prompt, and choose whether to use FLAN-T5 or GPT-2 for generation.",
)

# Launch Gradio interface
iface.launch()
