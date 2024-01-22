from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
DialoGPT (dialogue generative pre-trained transformer)
Trained on 147M conversation-like exchanges extracted from Reddit comment chains over a period spanning from 2005 through 2017,

Example :
User:Does money buy happiness?
DialoGPT: Money buys happiness, but it also buys a lot of things that make you happy.
>> User:What is the best way to buy happiness ?
DialoGPT: Money. Money buys happiness.
>> User:What about gold 
DialoGPT: Gold is a good investment.
"""
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines
for step in range(3):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
