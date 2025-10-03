"""Main CLI entry point for VLM PEFT SMILES."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from src.dataset import load_dataset
from src.vlm_inference import VLMPredictor


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def evaluate_single(
    model_name: str,
    image_path: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    verbose: bool = False
):
    """Evaluate a single image with the VLM."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model: {model_name}")
    predictor = VLMPredictor(model_name=model_name)
    
    logger.info(f"Processing image: {image_path}")
    result = predictor.predict_single(
        image_paths=image_path,
        text_prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    print(f"Result: {result}")
    return result


def evaluate_batch(
    model_name: str,
    image_dir: str,
    prompt: str,
    output_file: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    batch_size: Optional[int] = None,
    verbose: bool = False
):
    """Evaluate all images in a directory."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Find all image files
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        logger.error(f"Image directory does not exist: {image_dir}")
        sys.exit(1)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_paths = [
        str(p) for p in image_dir_path.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        logger.error(f"No image files found in: {image_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Create prompts
    prompts = [prompt] * len(image_paths)
    
    # Initialize predictor and run batch inference
    logger.info(f"Loading model: {model_name}")
    predictor = VLMPredictor(model_name=model_name)
    
    results = predictor.batch_predict_from_dataset(
        image_paths=[path for path in image_paths],  # Cast to proper type
        text_prompts=prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size
    )
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            for img_path, result in zip(image_paths, results):
                f.write(f"{img_path}\t{result}\n")
        logger.info(f"Results saved to: {output_file}")
    else:
        for img_path, result in zip(image_paths, results):
            print(f"{Path(img_path).name}: {result}")
    
    return results


def evaluate_dataset(
    model_name: str,
    dataset_path: str,
    prompt: str,
    output_file: Optional[str] = None,
    csv_output: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    verbose: bool = False
):
    """Evaluate a tar-based molecular dataset."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
        logger.info(f"Loaded dataset: {dataset.info}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)
    
    # Initialize predictor
    logger.info(f"Loading model: {model_name}")
    predictor = VLMPredictor(model_name=model_name)
    
    # Run evaluation
    evaluation_report = predictor.evaluate_dataset(
        dataset=dataset,
        text_prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size,
        limit=limit,
        csv_output=csv_output
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {evaluation_report['dataset_name']}")
    print(f"Model: {evaluation_report['model_name']}")
    print(f"Total samples: {evaluation_report['total_samples']}")
    print(f"Exact match accuracy: {evaluation_report['exact_match_accuracy']:.2%}")
    print(f"Canonical SMILES accuracy: {evaluation_report['canonical_smiles_accuracy']:.2%}")
    print(f"Correct predictions: {evaluation_report['correct_predictions']}")
    print(f"Incorrect predictions: {evaluation_report['incorrect_predictions']}")
    
    if csv_output:
        print(f"\n✓ Detailed results saved to: {csv_output}")
        print(f"  Columns: molecule_id, smiles_ground_truth, smiles_predicted, exact_match, canonical_match, ...")
    
    # Show some examples
    if evaluation_report['correct_examples']:
        print(f"\nSample correct predictions:")
        for i, example in enumerate(evaluation_report['correct_examples'][:2]):
            print(f"  {i+1}. GT: {example['ground_truth']} | Pred: {example['prediction']}")
    
    if evaluation_report['incorrect_examples']:
        print(f"\nSample incorrect predictions:")
        for i, example in enumerate(evaluation_report['incorrect_examples'][:2]):
            print(f"  {i+1}. GT: {example['ground_truth']} | Pred: {example['prediction']}")
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w') as f:
            # Remove non-serializable objects for JSON
            json_report = {
                k: v for k, v in evaluation_report.items() 
                if k not in ['molecules', 'results_dataframe']  # Skip complex objects
            }
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON summary saved to: {output_file}")
    
    return evaluation_report


def explore_dataset(
    dataset_path: str,
    verbose: bool = False
):
    """Explore dataset structure and content."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)
    
    print("="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    print(f"Dataset: {dataset.info.dataset_name}")
    print(f"Total molecules: {dataset.info.total_molecules}")
    print(f"Number of shards: {dataset.info.num_shards}")
    print(f"Render config: {dataset.info.render_config}")
    
    # Show some sample molecules
    print(f"\nSample molecules (first 5):")
    sample_molecules = dataset.get_sample(limit=5)
    for i, mol in enumerate(sample_molecules):
        print(f"  {i+1}. {mol}")
    
    # Show unique SMILES count
    unique_smiles = dataset.get_unique_smiles()
    print(f"\nUnique SMILES: {len(unique_smiles)}")
    
    # Show frequency distribution
    from collections import Counter
    smiles_counts = Counter(mol.smiles for mol in dataset.get_molecules())
    freq_dist = Counter(smiles_counts.values())
    
    print(f"\nFrequency distribution:")
    for freq in sorted(freq_dist.keys()):
        count = freq_dist[freq]
        print(f"  Frequency {freq}: {count} molecules")
    
    return dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VLM PEFT SMILES - Vision Language Model evaluation and fine-tuning"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single image evaluation
    single_parser = subparsers.add_parser('single', help='Evaluate a single image')
    single_parser.add_argument('image_path', help='Path to the image file')
    single_parser.add_argument('prompt', help='Text prompt for the model')
    single_parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct',
                              help='Model name or path (default: Qwen/Qwen2.5-VL-3B-Instruct)')
    single_parser.add_argument('--system-prompt', help='System prompt to guide model behavior')
    single_parser.add_argument('--temperature', type=float, default=0.0,
                              help='Sampling temperature (default: 0.0)')
    single_parser.add_argument('--max-tokens', type=int, default=512,
                              help='Maximum tokens to generate (default: 512)')
    single_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Enable verbose logging')
    
    # Batch evaluation
    batch_parser = subparsers.add_parser('batch', help='Evaluate all images in a directory')
    batch_parser.add_argument('image_dir', help='Directory containing image files')
    batch_parser.add_argument('prompt', help='Text prompt for the model')
    batch_parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct',
                             help='Model name or path (default: Qwen/Qwen2.5-VL-3B-Instruct)')
    batch_parser.add_argument('--output', '-o', help='Output file to save results')
    batch_parser.add_argument('--system-prompt', help='System prompt to guide model behavior')
    batch_parser.add_argument('--temperature', type=float, default=0.0,
                             help='Sampling temperature (default: 0.0)')
    batch_parser.add_argument('--max-tokens', type=int, default=512,
                             help='Maximum tokens to generate (default: 512)')
    batch_parser.add_argument('--batch-size', type=int,
                             help='Process in smaller batches (optional)')
    batch_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose logging')
    
    # Dataset evaluation
    dataset_parser = subparsers.add_parser('dataset', help='Evaluate a tar-based molecular dataset')
    dataset_parser.add_argument('dataset_path', help='Path to directory containing manifest.json')
    dataset_parser.add_argument('prompt', help='Text prompt for the model')
    dataset_parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct',
                               help='Model name or path (default: Qwen/Qwen2.5-VL-3B-Instruct)')
    dataset_parser.add_argument('--output', '-o', help='Output JSON file to save summary results')
    dataset_parser.add_argument('--csv', help='Output CSV file to save detailed predictions for analysis')
    dataset_parser.add_argument('--system-prompt', help='System prompt to guide model behavior')
    dataset_parser.add_argument('--temperature', type=float, default=0.0,
                               help='Sampling temperature (default: 0.0)')
    dataset_parser.add_argument('--max-tokens', type=int, default=512,
                               help='Maximum tokens to generate (default: 512)')
    dataset_parser.add_argument('--batch-size', type=int,
                               help='Process in smaller batches (optional)')
    dataset_parser.add_argument('--limit', type=int,
                               help='Maximum number of molecules to evaluate (for testing)')
    dataset_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose logging')
    
    # Dataset exploration (new command)
    explore_parser = subparsers.add_parser('explore', help='Explore dataset structure and content')
    explore_parser.add_argument('dataset_path', help='Path to directory containing manifest.json')
    explore_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'single':
        evaluate_single(
            model_name=args.model,
            image_path=args.image_path,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose
        )
    elif args.command == 'batch':
        evaluate_batch(
            model_name=args.model,
            image_dir=args.image_dir,
            prompt=args.prompt,
            output_file=args.output,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
    elif args.command == 'dataset':
        evaluate_dataset(
            model_name=args.model,
            dataset_path=args.dataset_path,
            prompt=args.prompt,
            output_file=args.output,
            csv_output=getattr(args, 'csv', None),
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            limit=args.limit,
            verbose=args.verbose
        )
    elif args.command == 'explore':
        explore_dataset(
            dataset_path=args.dataset_path,
            verbose=args.verbose
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
