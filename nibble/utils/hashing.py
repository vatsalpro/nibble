"""
Model hashing and checksum verification utility.
Computes SHA-256 digests for model artifacts and files to guarantee reproducibility.
"""

import hashlib
from pathlib import Path


def calculate_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hexadecimal digest for a file."""
    path = Path(file_path)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
