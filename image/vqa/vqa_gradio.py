from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import json
import urllib.request
import numpy as np
import gradio as gr

def logit_to_percentage(logit):
    probability = 1 / (1 + np.exp(-logit.detach().numpy()))
    percentage = probability * 100
    return percentage

def vqa_inference(image, question):
    # Load ViltProcessor and ViltForQuestionAnswering
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # Preprocess inputs
    encoding = processor(image, question, return_tensors="pt")

    # Forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    
    # Get the predicted answer and confidence
    predicted_answer = model.config.id2label[idx]
    confidence = logit_to_percentage(logits[0])[idx]

    return f"Predicted answer: {predicted_answer}", f"Confidence: {confidence:.2f}%"

# Define the Gradio interface
iface = gr.Interface(
    fn=vqa_inference,
    inputs=[

        gr.Image(type="pil", label="Input Image"),

        gr.Textbox(label="Question", lines=5),


    ],
    outputs=[
        gr.Textbox( label="Answer"),
        gr.Textbox( label="Confidence"),
    ],
    live=True,
)

# Launch the Gradio interface
iface.launch()
