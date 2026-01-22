#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from openai import OpenAI

# Initialize OpenAI client (API key from environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text(prompt, max_tokens=150, temperature=0.7):
    """
    Generate text using GPT.
    
    Args:
        prompt (str): Input text prompt
        max_tokens (int): Length of generated text
        temperature (float): Creativity level (0.0 - 1.0)
    
    Returns:
        str: Generated text
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=max_tokens,
        temperature=temperature
    )

    return response.output_text



print("GPT-Based Text Generator\n")

prompt = input("Enter prompt:\n")
max_tokens = int(input("Enter max tokens (e.g. 100): "))
temperature = float(input("Enter creativity (0.0 - 1.0): "))

result = generate_text(prompt, max_tokens, temperature)

print("\nGenerated Text:\n")
print(result)




