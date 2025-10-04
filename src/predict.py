"""Prediction script for generating SMILES from molecular images using vLLM."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

from dataset import SMILESDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_system_prompt(prompt_path: str | None = None) -> str:
    """
    Load system prompt from file or return default.

    Args:
        prompt_path: Path to system prompt file. If None, uses default.

    Returns:
        System prompt text.
    """
    if prompt_path is None:
        # Use default prompt file relative to this module
        default_path = (
            Path(__file__).parent.parent / "prompts" / "default_system_prompt.txt"
        )
        prompt_path = str(default_path)

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {prompt_path}")
        # Fallback to a basic prompt
        return (
            "You are an expert chemist. Analyze the molecular structure image and "
            "provide only the SMILES notation without any additional text."
        )


def predict_dataset(
    dataset: SMILESDataset,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    text_prompt: str = "What is the SMILES representation of this molecule?",
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_model_len: int = 1024,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Generate predictions for a dataset and return results as a DataFrame.

    Args:
        dataset: SMILESDataset instance.
        model_name: Model name or path for vLLM.
        text_prompt: Question/instruction to ask about the image.
        system_prompt: System prompt to guide model behavior.
        temperature: Sampling temperature (0.0 for greedy).
        max_tokens: Maximum tokens to generate.
        max_model_len: Maximum sequence length for the model.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        limit: Maximum number of samples to process (None for all).

    Returns:
        DataFrame with 'ground_truth' and 'prediction' columns.
    """
    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {model_name}")
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )

    # Determine how many samples to process
    num_samples = len(dataset) if limit is None else min(limit, len(dataset))
    logger.info(f"Processing {num_samples} samples from dataset")

    # Prepare data
    ground_truth_smiles = []
    images = []

    logger.info("Loading dataset samples...")
    for i in range(num_samples):
        smiles, image = dataset[i]
        ground_truth_smiles.append(smiles)
        images.append(image)

        if (i + 1) % 1000 == 0:
            logger.info(f"Loaded {i + 1}/{num_samples} samples")

    # Prepare prompts for vLLM
    logger.info("Preparing prompts for inference...")
    vllm_prompts = []

    for image in images:
        # Format the text prompt for Qwen2.5-VL with correct vision tokens
        if system_prompt:
            formatted_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            formatted_text = (
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
            )

        # Create prompt dict for vLLM
        vllm_prompt = {
            "prompt": formatted_text,
            "multi_modal_data": {"image": image},
        }
        vllm_prompts.append(vllm_prompt)

    # Run batch inference
    logger.info(f"Running batch inference on {len(vllm_prompts)} prompts...")
    outputs = llm.generate(vllm_prompts, sampling_params=sampling_params)

    # Extract predictions
    predictions = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        predictions.append(generated_text)

    logger.info(f"Generated {len(predictions)} predictions")

    # Create DataFrame
    results_df = pd.DataFrame(
        {
            "ground_truth": ground_truth_smiles,
            "prediction": predictions,
        }
    )

    return results_df


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description="Generate SMILES predictions from molecular images using vLLM"
    )

    # Dataset arguments
    parser.add_argument(
        "--smiles-file",
        type=str,
        required=True,
        help="Path to text file containing SMILES strings (one per line)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("WIDTH", "HEIGHT"),
        help="Size of generated images (default: 224 224)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path for vLLM (default: Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum sequence length for the model (default: 1024)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)",
    )

    # Prompt arguments
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="What is the SMILES representation of this molecule?",
        help="Question/instruction to ask about the image",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Path to system prompt file (default: uses prompts/default_system_prompt.txt)",
    )

    # Sampling arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output CSV (default: auto-generated with timestamp)",
    )

    args = parser.parse_args()

    # Load system prompt
    system_prompt = load_system_prompt(args.system_prompt_file)
    logger.info("System prompt loaded")

    # Create dataset
    logger.info(f"Loading dataset from {args.smiles_file}")
    dataset = SMILESDataset(
        smiles_file=args.smiles_file,
        img_size=tuple(args.img_size),
        skip_invalid=True,
    )

    # Run predictions
    results_df = predict_dataset(
        dataset=dataset,
        model_name=args.model_name,
        text_prompt=args.text_prompt,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit=args.limit,
    )

    # Determine output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.split("/")[-1]
        output_path = f"predictions_{model_short}_{timestamp}.csv"
    else:
        output_path = args.output

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Total predictions: {len(results_df)}")
    logger.info(f"  Output file: {output_path}")

    # Quick accuracy check (exact match)
    exact_matches = (results_df["ground_truth"] == results_df["prediction"]).sum()
    accuracy = exact_matches / len(results_df) * 100
    logger.info(f"  Exact match accuracy: {exact_matches}/{len(results_df)} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
