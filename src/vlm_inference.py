"""vLLM-based inference module for Vision Language Models."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams

from .dataset import MoleculeEntry, TarShardDataset

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
    except Exception as e:
        logger.warning(f"Error reading system prompt file {prompt_path}: {e}")
        return (
            "You are an expert chemist. Analyze the molecular structure image and "
            "provide only the SMILES notation without any additional text."
        )


class VLMPredictor:
    """
    A class for batch inference with Vision Language Models using vLLM.

    This class provides efficient batch prediction capabilities for VLMs,
    specifically optimized for the Qwen2.5-VL series of models.

    Attributes:
        model_name: The name or path of the model to use.
        llm: The vLLM LLM instance.
        sampling_params: Default sampling parameters for generation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len: int = 1024,
        limit_mm_per_prompt: dict[str, int] | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        """
        Initialize the VLM predictor with vLLM.

        Args:
            model_name: The name or path of the model to load.
            max_model_len: Maximum sequence length for the model.
            limit_mm_per_prompt: Dictionary specifying limits for multimodal inputs
                                (e.g., {"image": 1, "video": 0}).
            tensor_parallel_size: Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0).
            **kwargs: Additional arguments to pass to vLLM's LLM constructor.
        """
        self.model_name = model_name

        # Set default multimodal limits if not provided
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 10}  # Support up to 10 images per prompt

        logger.info(f"Initializing vLLM with model: {model_name}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"Multimodal limits: {limit_mm_per_prompt}")

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            limit_mm_per_prompt=limit_mm_per_prompt,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )

        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding by default
            max_tokens=512,
            top_p=1.0,
        )

        logger.info("vLLM initialization complete")

    def create_message_with_images(
        self,
        image_paths: str | list[str],
        text_prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Create messages with images for Qwen2.5-VL models using the correct format.

        Args:
            image_paths: Single image path or list of image paths.
            text_prompt: The text instruction/question.
            system_prompt: Optional system prompt to set behavior.

        Returns:
            List of messages in the correct format.
        """
        # Normalize to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Build messages
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with images
        content = []

        # Add images first
        for image_path in image_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
            )

        # Add text
        content.append({"type": "text", "text": text_prompt})

        messages.append({"role": "user", "content": content})

        return messages

    def predict(
        self,
        prompts: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **sampling_kwargs,
    ) -> list[str]:
        """
        Perform batch prediction on a list of prompts.

        Args:
            prompts: List of prompt dictionaries with image paths and text.
            temperature: Sampling temperature (0.0 for greedy).
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            List of generated text responses.
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **sampling_kwargs,
        )

        logger.info(f"Running batch inference on {len(prompts)} prompts")

        # Prepare prompt dictionaries for vLLM generate method
        vllm_prompts = []

        for prompt in prompts:
            # Extract image paths and text
            image_paths = prompt.get("image_paths", [])
            text_prompt = prompt.get("text_prompt", "")
            system_prompt = prompt.get("system_prompt", "")

            # Format the text prompt for Qwen2.5-VL with correct vision tokens
            if system_prompt:
                formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

            # Load images
            images = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path)
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")

            # Create prompt dict for vLLM (not TextPrompt object)
            vllm_prompt = {
                "prompt": formatted_text,
                "multi_modal_data": {
                    "image": images[0] if len(images) == 1 else images
                },
            }
            vllm_prompts.append(vllm_prompt)

        # Run inference
        outputs = self.llm.generate(vllm_prompts, sampling_params=sampling_params)

        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        logger.info(f"Batch inference complete. Generated {len(results)} responses")

        return results

    def predict_single(
        self,
        image_paths: str | list[str],
        text_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **sampling_kwargs,
    ) -> str:
        """
        Convenience method for single prediction.

        Args:
            image_paths: Single image path or list of image paths.
            text_prompt: The text instruction/question.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            Generated text response.
        """
        # Normalize to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        prompt = {
            "image_paths": image_paths,
            "text_prompt": text_prompt,
            "system_prompt": system_prompt,
        }

        results = self.predict(
            [prompt], temperature=temperature, max_tokens=max_tokens, **sampling_kwargs
        )
        return results[0]

    def batch_predict_from_dataset(
        self,
        image_paths: list[str | list[str]],
        text_prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        batch_size: int | None = None,
        **sampling_kwargs,
    ) -> list[str]:
        """
        Batch predict from parallel lists of images and prompts.

        Args:
            image_paths: List of image paths (can be single path or list per item).
            text_prompts: List of text prompts.
            system_prompt: Optional system prompt to use for all.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            batch_size: If specified, process in smaller batches (not necessary with vLLM).
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            List of generated text responses.
        """
        assert len(image_paths) == len(text_prompts), (
            "Number of image paths must match number of text prompts"
        )

        # Create all prompts
        prompts = []
        for imgs, txt in zip(image_paths, text_prompts):
            # Normalize to list
            if isinstance(imgs, str):
                imgs = [imgs]

            prompt = {
                "image_paths": imgs,
                "text_prompt": txt,
                "system_prompt": system_prompt,
            }
            prompts.append(prompt)

        # vLLM handles batching efficiently, but allow manual batching if requested
        if batch_size is not None and batch_size < len(prompts):
            logger.info(f"Processing in batches of {batch_size}")
            all_results = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                results = self.predict(
                    batch,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **sampling_kwargs,
                )
                all_results.extend(results)
            return all_results
        else:
            # Process all at once
            return self.predict(
                prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                **sampling_kwargs,
            )

    def evaluate_dataset(
        self,
        dataset: TarShardDataset,
        text_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        batch_size: int | None = None,
        limit: int | None = None,
        cleanup_temp_files: bool = True,
        csv_output: str | None = None,
        **sampling_kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the model on a TarShardDataset and return predictions with ground truth.

        Args:
            dataset: TarShardDataset instance to evaluate on.
            text_prompt: The text prompt to use for all molecules.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            batch_size: Batch size for processing.
            limit: Maximum number of molecules to evaluate (for testing).
            cleanup_temp_files: Whether to clean up temporary image files.
            csv_output: Path to save results as CSV for later analysis.
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            Dictionary containing predictions, ground truth, and evaluation metrics.
        """
        # Get molecules to evaluate
        molecules = dataset.get_sample(limit=limit, random_seed=42)
        logger.info(f"Evaluating on {len(molecules)} molecules")

        # Extract images to temporary files
        logger.info("Extracting images to temporary files...")
        temp_paths = dataset.extract_images_to_temp(molecules)

        try:
            # Prepare input data
            image_paths = [
                temp_paths[mol.filename]
                for mol in molecules
                if mol.filename in temp_paths
            ]
            text_prompts = [text_prompt] * len(image_paths)
            ground_truth = [
                mol.smiles for mol in molecules if mol.filename in temp_paths
            ]

            # Adjust molecules list to match available images
            available_molecules = [
                mol for mol in molecules if mol.filename in temp_paths
            ]

            if len(image_paths) != len(molecules):
                logger.warning(
                    f"Only {len(image_paths)}/{len(molecules)} images were successfully extracted"
                )

            # Run batch prediction
            logger.info("Running batch inference...")
            predictions = self.batch_predict_from_dataset(
                image_paths=[path for path in image_paths],  # Cast to proper type
                text_prompts=text_prompts,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                **sampling_kwargs,
            )

            # Create detailed DataFrame for CSV export
            results_df = pd.DataFrame(
                {
                    "molecule_id": [mol.original_id for mol in available_molecules],
                    "smiles_ground_truth": ground_truth,
                    "smiles_predicted": predictions,
                    "filename": [mol.filename for mol in available_molecules],
                    "shard_file": [mol.shard_file for mol in available_molecules],
                    "model_name": [self.model_name] * len(predictions),
                    "text_prompt": [text_prompt] * len(predictions),
                    "system_prompt": [system_prompt] * len(predictions),
                    "temperature": [temperature] * len(predictions),
                    "max_tokens": [max_tokens] * len(predictions),
                    "timestamp": [datetime.now().isoformat()] * len(predictions),
                }
            )

            # Add correctness flags
            results_df["exact_match"] = (
                results_df["smiles_ground_truth"] == results_df["smiles_predicted"]
            )

            # Try to add canonical SMILES comparison if RDKit is available
            try:
                from rdkit import Chem

                def canonicalize_smiles(smiles):
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        return Chem.MolToSmiles(mol) if mol else smiles
                    except:
                        return smiles

                results_df["smiles_ground_truth_canonical"] = results_df[
                    "smiles_ground_truth"
                ].apply(canonicalize_smiles)
                results_df["smiles_predicted_canonical"] = results_df[
                    "smiles_predicted"
                ].apply(canonicalize_smiles)
                results_df["canonical_match"] = (
                    results_df["smiles_ground_truth_canonical"]
                    == results_df["smiles_predicted_canonical"]
                )
                logger.info("Added canonical SMILES comparison using RDKit")
            except ImportError:
                logger.warning("RDKit not available. Falling back to exact match only.")
                results_df["canonical_match"] = results_df["exact_match"]

            # Save CSV if requested
            if csv_output:
                csv_path = Path(csv_output)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(csv_path, index=False)
                logger.info(f"Results saved to CSV: {csv_path}")

            # Calculate metrics
            from .dataset import DatasetEvaluator

            report = DatasetEvaluator.create_evaluation_report(
                predictions=predictions,
                ground_truth=ground_truth,
                molecules=available_molecules,
            )

            # Add additional info
            report.update(
                {
                    "dataset_name": dataset.info.dataset_name,
                    "model_name": self.model_name,
                    "text_prompt": text_prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "molecules": available_molecules,
                    "results_dataframe": results_df,
                    "csv_output": csv_output,
                }
            )

            return report

        finally:
            # Clean up temporary files
            if cleanup_temp_files:
                logger.info("Cleaning up temporary files...")
                for temp_path in temp_paths.values():
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_path}: {e}")

    def predict_molecules(
        self,
        molecules: list[MoleculeEntry],
        dataset: TarShardDataset,
        text_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        cleanup_temp_files: bool = True,
        **sampling_kwargs,
    ) -> list[str]:
        """
        Predict SMILES for a specific list of molecules.

        Args:
            molecules: List of MoleculeEntry objects to predict.
            dataset: TarShardDataset instance for loading images.
            text_prompt: The text prompt to use.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            cleanup_temp_files: Whether to clean up temporary files.
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            List of predicted SMILES strings.
        """
        logger.info(f"Predicting SMILES for {len(molecules)} molecules")

        # Extract images to temporary files
        temp_paths = dataset.extract_images_to_temp(molecules)

        try:
            # Prepare input data
            image_paths = [
                temp_paths[mol.filename]
                for mol in molecules
                if mol.filename in temp_paths
            ]
            text_prompts = [text_prompt] * len(image_paths)

            if len(image_paths) != len(molecules):
                logger.warning(
                    f"Only {len(image_paths)}/{len(molecules)} images were successfully extracted"
                )

            # Run prediction
            predictions = self.batch_predict_from_dataset(
                image_paths=[path for path in image_paths],  # Cast to proper type
                text_prompts=text_prompts,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **sampling_kwargs,
            )

            return predictions

        finally:
            # Clean up temporary files
            if cleanup_temp_files:
                for temp_path in temp_paths.values():
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_path}: {e}")
