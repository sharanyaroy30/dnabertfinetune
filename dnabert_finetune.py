#!/usr/bin/env python3
"""
DNABERT-2 Fine-tuning Script for Genomic Sequence Classification

This script fine-tunes the DNABERT-2 model for binary classification of genomic sequences.
It expects a CSV file with genomic coordinates and creates sequences around those coordinates.

Usage:
    python dnabert_finetune.py --csv_file your_data.csv [--output_dir ./output]
"""

import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def extract_sequence(center, window=75, mock_genome=None):
    """
    Extract genomic sequence around a center position.
    
    Args:
        center: Center position for sequence extraction
        window: Half-window size (default 75bp for 151bp total)
        mock_genome: Mock genome string (replace with actual genome data)
    
    Returns:
        Extracted sequence string
    """
    if mock_genome is None:
        # Default mock genome - replace with actual genome FASTA loading
        mock_genome = "ACGT" * 10**6  # 4 million bases, example only
    
    try:
        center = int(center)
        start = max(0, center - window - 1)  # 0-based indexing
        end = start + 2*window + 1
        return mock_genome[start:end]
    except:
        return ""  # empty string for invalid positions

def load_and_preprocess_data(csv_file):
    """
    Load CSV data and preprocess for DNABERT-2 training.
    
    Args:
        csv_file: Path to CSV file with genomic data
    
    Returns:
        Preprocessed dataset ready for training
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print("Extracting sequences...")
    # Assuming 'Unnamed: 2' column contains genomic coordinates
    # Adjust this based on your actual CSV structure
    df['sequence'] = df['Unnamed: 2'].apply(extract_sequence)
    
    # Create binary labels - adjust this condition based on your needs
    df['label'] = (df['Unnamed: 2'] > 100000).astype(int)
    
    # Filter out empty sequences
    df_clean = df[['sequence', 'label']].dropna()
    df_clean = df_clean[df_clean['sequence'] != '']
    
    print(f"Processed {len(df_clean)} valid sequences")
    print(f"Label distribution: {df_clean['label'].value_counts().to_dict()}")
    
    return Dataset.from_pandas(df_clean)

def tokenize_function(examples, tokenizer):
    """Tokenize sequences for DNABERT-2."""
    return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=151)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune DNABERT-2 for genomic sequence classification')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with genomic data')
    parser.add_argument('--output_dir', type=str, default='./dnabert_finetuned', help='Output directory for trained model')
    parser.add_argument('--model_name', type=str, default='zhihan1996/DNABERT-2-117M', help='DNABERT-2 model name')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found!")
        return
    
    # Load and preprocess data
    dataset = load_and_preprocess_data(args.csv_file)
    
    # Load DNABERT-2 tokenizer and model
    print(f"Loading DNABERT-2 model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, 
        trust_remote_code=True
    )
    
    # Tokenize dataset
    print("Tokenizing sequences...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True
    )
    
    # Split dataset (80% train, 20% eval)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=10,
        push_to_hub=False,
        report_to="none",  # disable wandb logging
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    final_model_path = args.output_dir + "_final"
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    
    # Print training summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final model saved to: {final_model_path}")
    print(f"Training logs saved to: {os.path.join(args.output_dir, 'logs')}")
    print("\nTo load the trained model:")
    print(f"from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    print(f"model = AutoModelForSequenceClassification.from_pretrained('{final_model_path}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{final_model_path}')")

if __name__ == "__main__":
    main()