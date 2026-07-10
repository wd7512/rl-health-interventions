"""NormWear foundation model wrapper for FM embedding extraction.

Loads the pretrained NormWear model from HuggingFace and extracts
encoder embeddings from wearable sensor signals.

Model: mosaic-laboratory/normwear (Apache 2.0)
Input: [batch, channels, sequence_length] sensor tensor
Output: [CLS] embedding vector (768-dim)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# NormWear output embedding dimension
NORMWEAR_EMBED_DIM = 768

# Model ID on HuggingFace
MODEL_ID = "mosaic-laboratory/normwear"


class NormWearEncoder:
    """Wraps NormWear for extracting health-state embeddings.

    Usage:
        encoder = NormWearEncoder()
        embedding = encoder.encode(sensor_tensor)
        # embedding: np.ndarray of shape [768]
    """

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        logger.info("Loading NormWear from %s (this may take a moment)...", MODEL_ID)

        # NormWear requires these torch.uint patches (see their README)
        torch.uint64 = torch.int64
        torch.uint32 = torch.int32
        torch.uint16 = torch.int16

        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()
        logger.info("NormWear loaded on %s", self.device)

    @torch.no_grad()
    def encode(self, sensor_tensor: np.ndarray) -> np.ndarray:
        """Extract [CLS] embedding from sensor signals.

        Args:
            sensor_tensor: shape [batch, channels, sequence_length]
                          e.g. [1, 3, 256] for 1 sample, 3 sensors, 4s at 64Hz

        Returns:
            [CLS] embedding of shape [batch, 768]
        """
        self._load_model()
        assert self._model is not None

        x = torch.tensor(sensor_tensor, dtype=torch.float32).to(self.device)

        outpack = self._model(
            x,
            return_spec=True,
            return_enc_out=True,
            return_dec_out=False,
            zero_shot_input_pack=None,
        )

        # enc_out shape: [batch, nvar, num_patches, embed_size]
        # The [CLS] embedding is the 1st patch (index 0)
        enc_out = outpack["enc_out"]
        cls_embedding = enc_out[:, :, 0, :]  # [batch, nvar, embed_size]

        # Average across channels to get a single embedding per sample
        # Shape: [batch, embed_size]
        avg_embedding = cls_embedding.mean(dim=1)

        return avg_embedding.cpu().numpy().astype(np.float64)

    @property
    def embedding_dim(self) -> int:
        return NORMWEAR_EMBED_DIM
