import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import pipeline, set_seed

# Load models and tokenizers
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

gpt2_generator = pipeline('text-generation', model='gpt2')

dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def generate_response(input_text, selected_model):
    if selected_model == "FLAN-T5":
        model = flan_model
        tokenizer = flan_tokenizer
    elif selected_model == "GPT-2":
        model = gpt2_generator.model
        tokenizer = gpt2_generator.tokenizer
    else:
        model = dialogpt_model
        tokenizer = dialogpt_tokenizer

    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output
    outputs = model.generate(**inputs, max_length=100)

    # Decode and return the response
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded_output

# Interface for Gradio
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(),
        gr.Radio(["FLAN-T5", "GPT-2", "DialoGPT"],  label="Select Model"),
    ],
    outputs="text",
    live=False,
    title="Text Generation",
    description="Enter a prompt and choose the model for text generation.",
)

# Launch Gradio interface
iface.launch()
