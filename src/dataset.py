"""Dataset loading utilities for molecular image datasets."""

import io
import json
import logging
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MoleculeEntry:
    """Represents a single molecule entry in the dataset."""
    smiles: str
    filename: str
    original_id: int
    shard_file: str
    
    def __str__(self) -> str:
        return f"MoleculeEntry(smiles='{self.smiles}', filename='{self.filename}', id={self.original_id})"


@dataclass
class DatasetInfo:
    """Contains metadata about the dataset."""
    dataset_name: str
    total_molecules: int
    shard_size: int
    render_config: dict[str, Any]
    num_shards: int
    
    def __str__(self) -> str:
        return f"Dataset '{self.dataset_name}': {self.total_molecules} molecules in {self.num_shards} shards"


class TarShardDataset:
    """
    Handles loading molecular datasets stored in tar shards with manifest files.
    
    This class is designed to work with datasets that have:
    - A manifest.json file describing the dataset structure
    - One or more tar files containing the molecular images
    - SMILES annotations for each molecule
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the directory containing manifest.json and shard files.
        """
        self.dataset_path = Path(dataset_path)
        self.manifest_path = self.dataset_path / "manifest.json"
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        # Load manifest
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Parse dataset info
        self.info = DatasetInfo(
            dataset_name=self.manifest["dataset_name"],
            total_molecules=self.manifest["total_molecules"],
            shard_size=self.manifest["shard_size"],
            render_config=self.manifest["render_config"],
            num_shards=len(self.manifest["shards"])
        )
        
        # Parse molecules by shard
        self._molecules_by_shard = {}
        self._all_molecules = []
        
        for shard_info in self.manifest["shards"]:
            shard_number = shard_info["shard_number"]
            shard_file = shard_info["shard_file"]
            
            molecules = []
            for mol_data in shard_info["molecules"]:
                molecule = MoleculeEntry(
                    smiles=mol_data["smiles"],
                    filename=mol_data["filename"],
                    original_id=mol_data["original_id"],
                    shard_file=shard_file
                )
                molecules.append(molecule)
                self._all_molecules.append(molecule)
            
            self._molecules_by_shard[shard_number] = molecules
        
        logger.info(f"Loaded {self.info}")
        logger.info(f"Total molecules loaded: {len(self._all_molecules)}")
    
    def get_molecules(self, shard_number: int | None = None) -> list[MoleculeEntry]:
        """
        Get molecules from a specific shard or all molecules.
        
        Args:
            shard_number: If specified, return only molecules from this shard.
                         If None, return all molecules.
        
        Returns:
            List of MoleculeEntry objects.
        """
        if shard_number is not None:
            if shard_number not in self._molecules_by_shard:
                raise ValueError(f"Shard {shard_number} not found")
            return self._molecules_by_shard[shard_number]
        return self._all_molecules
    
    def load_image(self, molecule: MoleculeEntry) -> Image.Image:
        """
        Load the image for a specific molecule.
        
        Args:
            molecule: The MoleculeEntry to load the image for.
        
        Returns:
            PIL Image object.
        """
        shard_path = self.dataset_path / Path(molecule.shard_file)
        
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        
        with tarfile.open(shard_path, 'r') as tar:
            try:
                member = tar.getmember(molecule.filename)
                image_data = tar.extractfile(member).read()
                image = Image.open(io.BytesIO(image_data))
                return image
            except KeyError:
                raise FileNotFoundError(f"Image {molecule.filename} not found in {shard_path}")

    def extract_images_to_temp(self, molecules: list[MoleculeEntry]) -> dict[str, str]:
        """
        Extract images to temporary files and return a mapping of filename to temp path.
        
        Args:
            molecules: list[MoleculeEntry] to extract images for.

        Returns:
            Dictionary mapping molecule filename to temporary file path.
        """
        temp_paths = {}
        
        # Group molecules by shard for efficient extraction
        molecules_by_shard = {}
        for mol in molecules:
            shard_file = mol.shard_file
            if shard_file not in molecules_by_shard:
                molecules_by_shard[shard_file] = []
            molecules_by_shard[shard_file].append(mol)
        
        # Extract from each shard
        for shard_file, shard_molecules in molecules_by_shard.items():
            shard_path = self.dataset_path / shard_file
            
            with tarfile.open(shard_path, 'r') as tar:
                for mol in shard_molecules:
                    try:
                        member = tar.getmember(mol.filename)
                        image_data = tar.extractfile(member).read()
                        
                        # Create temporary file
                        suffix = Path(mol.filename).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                            temp_file.write(image_data)
                            temp_paths[mol.filename] = temp_file.name
                    
                    except KeyError:
                        logger.warning(f"Image {mol.filename} not found in {shard_path}")
                        continue
        
        return temp_paths

    def get_sample(self, limit: int | None = None, random_seed: int | None = None) -> list[MoleculeEntry]:
        """
        Get a sample of molecules from the dataset.
        
        Args:
            limit: Maximum number of molecules to return. If None, return all.
            random_seed: Random seed for reproducible sampling.
        
        Returns:
            List of sampled MoleculeEntry objects.
        """
        molecules = self._all_molecules.copy()
        
        if random_seed is not None:
            import random
            random.seed(random_seed)
            random.shuffle(molecules)
        
        if limit is not None:
            molecules = molecules[:limit]
        
        return molecules

    def iterate_batches(self, batch_size: int, shuffle: bool = False, random_seed: int | None = None) -> Iterator[list[MoleculeEntry]]:
        """
        Iterate over the dataset in batches.
        
        Args:
            batch_size: Number of molecules per batch.
            shuffle: Whether to shuffle the dataset before batching.
            random_seed: Random seed for reproducible shuffling.
        
        Yields:
            Batches of MoleculeEntry objects.
        """
        molecules = self._all_molecules.copy()
        
        if shuffle:
            import random
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(molecules)
        
        for i in range(0, len(molecules), batch_size):
            yield molecules[i:i + batch_size]
    
    def get_unique_smiles(self) -> list[str]:
        """Get list of unique SMILES strings in the dataset."""
        return list(set(mol.smiles for mol in self._all_molecules))
    
    def __len__(self) -> int:
        """Return the total number of molecules in the dataset."""
        return len(self._all_molecules)
    
    def __getitem__(self, index: int) -> MoleculeEntry:
        """Get a molecule by index."""
        return self._all_molecules[index]


class DatasetEvaluator:
    """Helper class for evaluating VLM predictions against ground truth SMILES."""
    
    @staticmethod
    def exact_match_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
        """
        Calculate exact match accuracy between predictions and ground truth.
        
        Args:
            predictions: List of predicted SMILES strings.
            ground_truth: List of ground truth SMILES strings.
        
        Returns:
            Accuracy as a float between 0 and 1.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip() == gt.strip())
        return correct / len(predictions)
    
    @staticmethod
    def canonical_smiles_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
        """
        Calculate accuracy using canonical SMILES comparison.
        Requires RDKit for SMILES canonicalization.
        
        Args:
            predictions: List of predicted SMILES strings.
            ground_truth: List of ground truth SMILES strings.
        
        Returns:
            Accuracy as a float between 0 and 1.
        """
        try:
            from rdkit import Chem
        except ImportError:
            logger.warning("RDKit not available. Falling back to exact match.")
            return DatasetEvaluator.exact_match_accuracy(predictions, ground_truth)
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            try:
                pred_mol = Chem.MolFromSmiles(pred.strip())
                gt_mol = Chem.MolFromSmiles(gt.strip())
                
                if pred_mol is not None and gt_mol is not None:
                    pred_canonical = Chem.MolToSmiles(pred_mol, canonical=True)
                    gt_canonical = Chem.MolToSmiles(gt_mol, canonical=True)
                    
                    if pred_canonical == gt_canonical:
                        correct += 1
            except Exception as e:
                logger.debug(f"Error canonicalizing SMILES: {e}")
                continue
        
        return correct / len(predictions)
    
    @staticmethod
    def create_evaluation_report(predictions: list[str], ground_truth: list[str], molecules: list[MoleculeEntry]) -> dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        
        Args:
            predictions: List of predicted SMILES strings.
            ground_truth: List of ground truth SMILES strings.
            molecules: List of corresponding MoleculeEntry objects.
        
        Returns:
            Dictionary containing evaluation metrics and details.
        """
        if len(predictions) != len(ground_truth) or len(predictions) != len(molecules):
            raise ValueError("All input lists must have the same length")
        
        exact_acc = DatasetEvaluator.exact_match_accuracy(predictions, ground_truth)
        canonical_acc = DatasetEvaluator.canonical_smiles_accuracy(predictions, ground_truth)
        
        # Find examples of correct and incorrect predictions
        correct_examples = []
        incorrect_examples = []
        
        for i, (pred, gt, mol) in enumerate(zip(predictions, ground_truth, molecules)):
            if pred.strip() == gt.strip():
                correct_examples.append({
                    "index": i,
                    "molecule_id": mol.original_id,
                    "filename": mol.filename,
                    "prediction": pred,
                    "ground_truth": gt
                })
            else:
                incorrect_examples.append({
                    "index": i,
                    "molecule_id": mol.original_id,
                    "filename": mol.filename,
                    "prediction": pred,
                    "ground_truth": gt
                })
        
        return {
            "total_samples": len(predictions),
            "exact_match_accuracy": exact_acc,
            "canonical_smiles_accuracy": canonical_acc,
            "correct_predictions": len(correct_examples),
            "incorrect_predictions": len(incorrect_examples),
            "correct_examples": correct_examples[:5],  # First 5 correct examples
            "incorrect_examples": incorrect_examples[:5],  # First 5 incorrect examples
        }


def load_dataset(dataset_path: str) -> TarShardDataset:
    """
    Convenience function to load a tar-based molecular dataset.
    
    Args:
        dataset_path: Path to directory containing manifest.json and shard files.
    
    Returns:
        TarShardDataset instance.
    """
    return TarShardDataset(dataset_path)