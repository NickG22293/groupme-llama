import argparse
import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")

    # Load dataset from all files in the msgs/ directory
    print("Loading dataset from msgs/ directory...")
    data_dir = "msgs/"
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]

    # Combine all files into a single list of texts
    combined_data = []
    for file in all_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            combined_data.extend(data)  # Assuming each file contains a list of {"text": "..."}

    # Write combined data to a temporary JSON file
    temp_file = "combined_data.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump({"text": [item["text"] for item in combined_data]}, f)

    # Load dataset from the temporary file
    dataset = load_dataset("json", data_files={"train": temp_file})

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)
        # return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length, padding="max_length")

    print("Tokenizing...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # Train
    print("Training the model...")
    trainer.train()

    # Save the model
    print("Saving the model...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
