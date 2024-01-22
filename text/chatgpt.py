import openai
from dotenv import load_dotenv
from dotenv import dotenv_values

import os


secrets = dotenv_values(".secret")
openai.api_key = secrets["API_KEY"]

def mychat_with_chatgpt(prompt, model="gpt-3.5-turbo"):
    reponse= openai.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message= reponse.choices[0].text.strip()
    return message
userPrompt="Write a summary of fasting."
mGPTReponse= mychat_with_chatgpt(prompt=userPrompt)
print(mGPTReponse)
