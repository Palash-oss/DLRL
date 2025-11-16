from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn


class MultimodalAutoencoder(nn.Module):
    def __init__(self, text_dim: int = 128, image_dim: int = 5, latent_dim: int = 16):
        super().__init__()
        input_dim = text_dim + image_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, text_dim),
        )
        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, image_dim),
        )
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.latent_dim = latent_dim

    def forward(self, text_vec: torch.Tensor, image_vec: torch.Tensor):
        combined = torch.cat([text_vec, image_vec], dim=1)
        latent = self.encoder(combined)
        text_recon = self.text_decoder(latent)
        image_recon = self.image_decoder(latent)
        return text_recon, image_recon, latent

    def encode(self, text_vec: torch.Tensor, image_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([text_vec, image_vec], dim=1)
        return self.encoder(combined)


class MultimodalAutoencoderService:
    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[torch.device] = None,
        text_dim: int = 128,
        image_dim: int = 5,
        latent_dim: int = 16,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_path = self.checkpoint_dir / "multimodal_autoencoder.pt"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.model: Optional[MultimodalAutoencoder] = None
        self.available = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            print("Multimodal autoencoder checkpoint not found; latent features disabled.")
            return
        try:
            self.model = MultimodalAutoencoder(
                text_dim=self.text_dim,
                image_dim=self.image_dim,
                latent_dim=self.latent_dim,
            )
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            print("✓ Multimodal autoencoder loaded")
        except Exception as exc:
            print(f"Failed to load multimodal autoencoder: {exc}")
            self.model = None

    def _prepare_tensor(self, values: Optional[Sequence[float]], target_dim: int) -> torch.Tensor:
        tensor = torch.zeros((1, target_dim), dtype=torch.float32, device=self.device)
        if values:
            clipped = list(values)[:target_dim]
            tensor[0, : len(clipped)] = torch.tensor(clipped, dtype=torch.float32, device=self.device)
        return tensor

    def get_latent(
        self,
        text_values: Optional[Sequence[float]],
        image_values: Optional[Sequence[float]],
    ) -> Optional[Sequence[float]]:
        if not self.available:
            return None
        text_tensor = self._prepare_tensor(text_values, self.text_dim)
        image_tensor = self._prepare_tensor(image_values, self.image_dim)
        with torch.no_grad():
            latent = self.model.encode(text_tensor, image_tensor)
        return latent.squeeze(0).cpu().tolist()
