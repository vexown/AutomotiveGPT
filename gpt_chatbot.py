import torch
import gpt

# Load the pre-trained model
model = gpt.GPTLanguageModel()
model.load_state_dict(torch.load('gpt_model.pth'))
model.to(gpt.device)
model.eval()

# Example custom context
custom_context = " "
encoded_context = torch.tensor(gpt.encode(custom_context), dtype=torch.long, device=gpt.device).unsqueeze(0)

# Generate text
max_new_tokens = 500
generated_tokens = model.generate(encoded_context, max_new_tokens)[0].tolist()
generated_text = gpt.decode(generated_tokens)

print("Custom Context: \n", custom_context)
print("Generated Text: \n", generated_text)
