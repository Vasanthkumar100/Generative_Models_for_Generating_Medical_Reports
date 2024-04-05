import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_medical_report(prompt_text, max_length=200):
    # Load pre-trained model and tokenizer
    model_name = "gpt2-medium"  # You can use other GPT variants
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate medical report
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, early_stopping=True)

    # Decode and return generated report
    generated_report = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_report

# Example usage
prompt_text = "Patient presented with cough, fever, and fatigue. "
generated_report = generate_medical_report(prompt_text)
print("Generated Medical Report:")
print(generated_report)
