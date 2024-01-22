from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import json
import urllib.request
import numpy as np
def logit_to_percentage(logit):
    probability = 1 / (1 + np.exp(-logit.detach().numpy()))
    percentage = probability * 100
    return percentage
# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("im1.jpg")

with urllib.request.urlopen(
    "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
) as url:
    id2ans = json.loads(url.read().decode())

#print(id2ans)
text = "what do you see ?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print(logit_to_percentage(logits[0])[idx])
#answer = id2ans[str(idx)]
#print("Predicted answer:", answer)
print("Predicted answer:", model.config.id2label[idx])
