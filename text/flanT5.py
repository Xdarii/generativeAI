from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

inputs = tokenizer("Does money buy happiness?", return_tensors="pt")
outputs = model.generate(**inputs,max_new_tokens=20)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))    
print("\n")

inputs = tokenizer("Does money buy happiness?,", return_tensors="pt")
outputs = model.generate(**inputs,max_new_tokens=20)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))  
print("\n")
inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
outputs = model.generate(**inputs,max_new_tokens=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))