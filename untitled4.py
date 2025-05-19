

!pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Converts text to tokens
model = GPT2LMHeadModel.from_pretrained("gpt2")    # Loads the GPT-2 model

prompt = "Life is like"  # Your input text
input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Convert text to numbers

# Generate text with sampling enabled
output = model.generate(
    input_ids,
    max_length=50,  # Max length of generated text
    num_return_sequences=2,  # Number of different outputs
    do_sample=True,  # Enables multiple sequences
    temperature=0.7,  # Controls randomness (0.7 = balanced)
    top_k=50,  # Filters unlikely words
    top_p=0.9,  # Uses nucleus sampling (better diversity)
    no_repeat_ngram_size=2,  # Avoids repeating phrases
)

# Print results properly
for i, seq in enumerate(output):
    print(f"{i+1}. {tokenizer.decode(seq, skip_special_tokens=True)}\n")