# VLM PEFT SMILES

A vLLM-based scaffolding for batch prediction and fine-tuning of Vision Language Models on molecular structure recognition tasks.

## Features

- **vLLM Integration**: Fast batch inference with Qwen2.5-VL-3B-Instruct
- **Dataset Support**: Efficient loading of tar-based molecular datasets 
- **CSV Caching**: Save detailed predictions for later analysis and rescoring
- **CLI Interface**: Comprehensive command-line interface for all operations
- **Evaluation Metrics**: Exact match and canonical SMILES accuracy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd vlm-peft-smiles

# Install dependencies with uv
uv sync
```

## Usage

### Dataset Exploration
Explore dataset structure and content:
```bash
uv run python main.py explore data/gdb11_size05
```

### Model Evaluation
Evaluate on a molecular dataset with CSV caching:
```bash
# Sample evaluation (3 molecules)
uv run python main.py dataset data/gdb11_size05 \
    "What is the SMILES representation of this molecular structure?" \
    --limit 3 --csv results_sample.csv

# Full evaluation with both CSV and JSON output
uv run python main.py dataset data/gdb11_size05 \
    "What is the SMILES representation of this molecular structure?" \
    --csv results_full.csv --output summary.json
```

### Single Image Evaluation
```bash
uv run python main.py single image.png "Describe this molecular structure"
```

### Batch Directory Evaluation
```bash
uv run python main.py batch images/ "What molecule is this?" --output results.txt
```

## CSV Analysis

Analyze cached results for detailed insights:
```bash
# Analyze a single result file
uv run python examples/analyze_results.py results_sample.csv

# Compare multiple models/settings
uv run python examples/analyze_results.py results1.csv results2.csv --compare
```

### CSV Output Format

The CSV files contain detailed information for each prediction:

| Column | Description |
|--------|-------------|
| `molecule_id` | Original molecule ID from dataset |
| `smiles_ground_truth` | True SMILES string |
| `smiles_predicted` | Model prediction |
| `exact_match` | Boolean: exact string match |
| `canonical_match` | Boolean: canonicalized SMILES match (requires RDKit) |
| `model_name` | Model used for prediction |
| `text_prompt` | Prompt sent to model |
| `system_prompt` | System prompt (if any) |
| `temperature` | Sampling temperature |
| `max_tokens` | Max tokens setting |
| `timestamp` | When prediction was made |
| `filename` | Image filename |
| `shard_file` | Source tar shard |

## Dataset Format

The system supports tar-based molecular datasets with:
- `manifest.json`: Dataset metadata and molecule listings
- `shard_XXXX.tar`: Compressed molecular images
- Automatic image extraction and temporary file management

Example dataset structure:
```
data/gdb11_size05/
├── manifest.json
└── shard_0000.tar
```

## Command Reference

```bash
# Main CLI commands
uv run python main.py explore <dataset_path>                    # Dataset exploration
uv run python main.py dataset <dataset_path> <prompt> [options] # Dataset evaluation  
uv run python main.py single <image_path> <prompt> [options]    # Single image
uv run python main.py batch <image_dir> <prompt> [options]      # Batch directory

# Analysis tools
uv run python examples/analyze_results.py <csv_file(s)> [--compare]
```

## Key Options

- `--csv`: Save detailed predictions as CSV for analysis
- `--output`: Save JSON summary of results  
- `--limit N`: Evaluate only N molecules (for testing)
- `--batch-size N`: Process in smaller batches
- `--temperature T`: Sampling temperature (0.0 for greedy)
- `--verbose`: Enable detailed logging

## Why This Structure?

**Consolidated CLI**: All functionality is now in `main.py` with clear subcommands (`explore`, `dataset`, `single`, `batch`) rather than scattered across multiple scripts.

**CSV Caching**: Unlike JSON summaries, CSV files allow for:
- Easy loading into pandas/Excel for analysis
- Rescoring with different metrics without re-running inference
- Comparison across different models/prompts
- Statistical analysis and visualization

**Separation of Concerns**:
- `main.py`: All inference and evaluation operations
- `examples/analyze_results.py`: Post-hoc analysis and comparison
- Clear data flow: inference → CSV cache → analysis

This structure supports both quick experimentation and rigorous analysis workflows.