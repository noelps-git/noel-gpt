from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load your writing
with open("my_writing.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters: {len(text):,}")

# Split into chunks of 500 characters each
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
print(f"Total chunks: {len(chunks)}")

# Load GPT-2 tokenizer and model
print("Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to(device)
print(f"GPT-2 parameters: {sum(p.numel() for p in model.parameters()):,}")

# Tokenize
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = Dataset.from_dict({"text": chunks})
tokenised = dataset.map(tokenize, batched=True)
tokenised = tokenised.train_test_split(test_size=0.1)
print(f"Train samples: {len(tokenised['train'])}")
print(f"Val samples: {len(tokenised['test'])}")

# Training settings
training_args = TrainingArguments(
    output_dir="./gosense-gpt-finetuned",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    learning_rate=2e-5,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised["train"],
    eval_dataset=tokenised["test"],
    data_collator=data_collator,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# Save
trainer.save_model("./gosense-gpt-final")
tokenizer.save_pretrained("./gosense-gpt-final")
print("Model saved.")