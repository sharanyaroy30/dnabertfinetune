# DNABERT-2 Fine-tuning for Genomic Sequence Classification

This repository contains a script to fine-tune the DNABERT-2 model for binary classification of genomic sequences.

## Problem Solved

The original error you encountered:
```
ModuleNotFoundError: No module named 'torch.distributed.device_mesh'
RuntimeError: Failed to import transformers.training_args
```

This was caused by version incompatibility between PyTorch and Transformers. The issue has been resolved by:
- Installing PyTorch 2.7.1+ which includes `torch.distributed.device_mesh`
- Installing compatible Transformers 4.54.1
- Setting up a proper virtual environment

## Setup Instructions

### 1. Activate the Virtual Environment
```bash
source venv/bin/activate
```

### 2. Verify Installation
```bash
python3 test_import.py
```

You should see:
```
✓ PyTorch version: 2.7.1+cu126
✓ torch.distributed.device_mesh imported successfully
✓ TrainingArguments imported successfully
✓ AutoTokenizer and AutoModelForSequenceClassification imported successfully
```

## Usage

### Basic Usage
```bash
source venv/bin/activate
python3 dnabert_finetune.py --csv_file your_data.csv
```

### Advanced Usage with Custom Parameters
```bash
source venv/bin/activate
python3 dnabert_finetune.py \
    --csv_file your_data.csv \
    --output_dir ./my_custom_output \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 3e-5
```

### Parameters
- `--csv_file`: Path to your CSV file with genomic data (required)
- `--output_dir`: Output directory for the trained model (default: ./dnabert_finetuned)
- `--model_name`: DNABERT-2 model variant (default: zhihan1996/DNABERT-2-117M)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-5)

## CSV Data Format

Your CSV file should contain:
- A column with genomic coordinates (currently assumes 'Unnamed: 2')
- The script will extract 151bp sequences (±75bp around each coordinate)
- Binary labels are created based on coordinate values (> 100000)

**Important**: You'll need to modify the following in `dnabert_finetune.py`:
1. Replace the mock genome with your actual genome FASTA data
2. Adjust the column name for genomic coordinates
3. Modify the labeling logic based on your classification task

## Example Workflow

1. **Prepare your data**: Ensure your CSV has genomic coordinates
2. **Activate environment**: `source venv/bin/activate`
3. **Run training**: `python3 dnabert_finetune.py --csv_file data.csv`
4. **Monitor progress**: Training logs will be saved in the output directory
5. **Use trained model**: Load the final model for predictions

## Output

The script will create:
- `dnabert_finetuned/`: Checkpoints during training
- `dnabert_finetuned_final/`: Final trained model
- Training logs and metrics

## Loading the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('./dnabert_finetuned_final')
tokenizer = AutoTokenizer.from_pretrained('./dnabert_finetuned_final')

# Example prediction
sequence = "ATCGATCGATCG..."  # Your 151bp sequence
inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding="max_length", max_length=151)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Troubleshooting

If you encounter import errors:
1. Make sure you're in the virtual environment: `source venv/bin/activate`
2. Run the test script: `python3 test_import.py`
3. If tests fail, reinstall packages: `pip install --force-reinstall torch transformers`

## Hardware Requirements

- GPU recommended for training (CUDA-enabled)
- At least 8GB RAM
- Sufficient disk space for model checkpoints (~2-3GB)

## Notes

- The script includes proper train/validation split (80/20)
- Evaluation is performed after each epoch
- Best model is saved based on evaluation loss
- All Colab-specific code has been removed for standalone use