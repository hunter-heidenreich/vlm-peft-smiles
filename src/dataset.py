"""PyTorch dataset for SMILES strings with molecular images."""

from pathlib import Path

import numpy as np

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    """Dataset that loads SMILES strings and generates molecular images using RDKit.

    Args:
        smiles_file: Path to text file containing SMILES strings (one per line)
        img_size: Size of generated images (width, height) in pixels
        skip_invalid: If True, skip invalid SMILES strings; if False, raise error
    """

    def __init__(
        self,
        smiles_file: str | Path,
        img_size: tuple[int, int] = (224, 224),
        skip_invalid: bool = True,
    ):
        self.smiles_file = Path(smiles_file)
        self.img_size = img_size
        self.skip_invalid = skip_invalid

        # Load and validate SMILES strings
        self.smiles_list = []
        self._load_smiles()

    def _load_smiles(self):
        """Load SMILES strings from file and optionally validate them."""
        seen = set()
        with open(self.smiles_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                smiles = line.strip()

                if not smiles:  # Skip empty lines
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    if not self.skip_invalid:
                        raise ValueError(f"Invalid SMILES at line {line_num}: {smiles}")
                    print(f"Warning: Skipping invalid SMILES at line {line_num}: {smiles}")
                    continue

                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

                # Add to list, avoiding duplicates
                if canonical_smiles not in seen:
                    self.smiles_list.append(canonical_smiles)
                    seen.add(canonical_smiles)

        print(f"Loaded {len(self.smiles_list)} SMILES strings from {self.smiles_file}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image]:
        """Get a SMILES string and its corresponding molecular image.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (smiles_string, pil_image)

        Raises:
            ValueError: If SMILES is invalid and skip_invalid is False
        """
        smiles = self.smiles_list[idx]

        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Generate molecular image
        img = Draw.MolToImage(mol, size=self.img_size)

        return smiles, img


def main():
    # Create dataset
    data_file = Path("data/gdb11_all0-1_8.txt")
    dataset = SMILESDataset(
        smiles_file=data_file, img_size=(224, 224), skip_invalid=False
    )

    print(f"Dataset size: {len(dataset)}")

    print("\n--- Example: Single item access ---")
    smiles, image = dataset[11000]
    print(f"SMILES: {smiles}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    image.save("sample_molecule.png")
    print(f"Saved image for SMILES '{smiles}' to sample_molecule.png")
    
    print("\n--- Example: Random subset access ---")
    n = 5
    ixs = np.random.choice(len(dataset), n, replace=False)
    for i, idx in enumerate(ixs, 1):
        smiles, image = dataset[idx]
        print(f"\nSample {i}:")
        print(f"  Index: {idx}")
        print(f"  SMILES: {smiles}")
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")
        image.save(f"sample_molecule_{i}.png")
        print(f"  Saved image to sample_molecule_{i}.png")


if __name__ == "__main__":
    main()
