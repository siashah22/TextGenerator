#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = 'gpt2'  # You can choose other versions like 'gpt2-medium', 'gpt2-large', etc.
model = TFGPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the tokenizer knows padding tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Function to generate text using GPT-2
def generate_text(model, tokenizer, prompt, max_length=500):
    # Encode input prompt to tensor
    input_ids = tokenizer.encode(prompt, return_tensors='tf')

    # Generate text from the input prompt
    output = model.generate(input_ids, 
                            max_length=max_length, 
                            num_return_sequences=1, 
                            no_repeat_ngram_size=2,  # Prevent repeating phrases
                            top_k=50,  # Sampling strategy
                            top_p=0.95,  # Sampling strategy
                            temperature=1.0,  # Sampling temperature
                            do_sample=True,  # Enable sampling
                            pad_token_id=tokenizer.pad_token_id)

    # Decode the generated text back to human-readable text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
prompt = "Once upon a time in a land far, far away"
generated_text = generate_text(model, tokenizer, prompt, max_length=200)

print("Generated Text:")
print(generated_text)


# In[ ]:





# In[ ]:




