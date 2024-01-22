"""
The model is trained in the Toronto Book Corpus.
It contains over 7,000 unique unpublished books from a variety of genres including Adventure,
Fantasy, and Romance. Crucially, it contains long stretches of contiguous text, which allows the
generative model to learn to condition on long-range information.
"""
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
output = generator("Does money buy happiness ?,", max_length=50, num_return_sequences=2)
print(output)
output = generator("Does money buy happiness ?", max_length=50, num_return_sequences=2)
print(output)
output = generator("A step by step recipe to make bolognese pasta:", max_length=300, num_return_sequences=1)
print(output)

