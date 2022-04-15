"""
test of Bert

Author: Tong
Time: 13-04-2021
"""

import torch
from transformers import BertModel, BertTokenizer

# Initialize the tokenizer with a pretrained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert the string "granola bars" to tokenized vocabulary IDs
tokenizer.add_tokens(["<t>", "<\t>"])
granola_ids = tokenizer.encode('He was the chairman, chief executive officer (CEO), and co-founder of Apple Inc.')

# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))
# Convert the IDs to the actual vocabulary item
# Notice how the subword unit (suffix) starts with "##" to indicate
# that it is part of the previous string
print('granola_tokens', tokenizer.convert_ids_to_tokens(granola_ids))

# Convert the list of IDs to a tensor of IDs
granola_ids = torch.LongTensor(granola_ids)
# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.resize_token_embeddings(len(tokenizer))


# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
granola_ids = granola_ids.to(device)

model.eval()

print(granola_ids.size())
# unsqueeze IDs to get batch size of 1 as added dimension
granola_ids = granola_ids.unsqueeze(0)
print(granola_ids.size())

print(type(granola_ids))
with torch.no_grad():
    out = model(input_ids=granola_ids)

# the output is a tuple
print(type(out))
# the tuple contains three elements as explained above)
print(len(out))
# we only want the hidden_states
hidden_states = out[2]
print(len(hidden_states))
