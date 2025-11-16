import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CLASS_LABELS = ("negative", "neutral", "positive")
TOKEN_PATTERN = re.compile(r"[\w']+")


class BiGRUSentiment(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, len(CLASS_LABELS))
        self.feat_dim = feat_dim

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.bigru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        if self.bigru.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits, last_hidden


class BiGRUTextEncoder:
    """Utility class that loads a pretrained Bi-GRU checkpoint if available."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[torch.device] = None,
        max_length: int = 160,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_path = self.checkpoint_dir / "text_bigru.pt"
        self.vocab_path = self.checkpoint_dir / "text_bigru_vocab.json"
        self.model: Optional[BiGRUSentiment] = None
        self.vocab: Optional[Dict[str, int]] = None
        self.available = False
        self._load()

    def _load(self) -> None:
        if not (self.model_path.exists() and self.vocab_path.exists()):
            print("Text Bi-GRU checkpoint not found; falling back to lexicon model only.")
            return
        try:
            with self.vocab_path.open("r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        except Exception as exc:
            print(f"Failed to load Bi-GRU vocabulary: {exc}")
            return
        try:
            self.model = BiGRUSentiment(len(self.vocab))
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            print("✓ Text Bi-GRU encoder loaded")
        except Exception as exc:
            print(f"Failed to load Bi-GRU checkpoint: {exc}")
            self.model = None

    def _tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())[: self.max_length]

    def _numericalize(self, tokens: List[str]) -> List[int]:
        if not self.vocab:
            return []
        unk_idx = self.vocab.get("<unk>", 1)
        pad_idx = self.vocab.get("<pad>", 0)
        ids = [self.vocab.get(tok, unk_idx) for tok in tokens]
        if len(ids) < self.max_length:
            ids.extend([pad_idx] * (self.max_length - len(ids)))
        else:
            ids = ids[: self.max_length]
        return ids

    def predict(self, text: Optional[str]) -> Optional[Dict]:
        if not self.available or not text:
            return None
        tokens = self._tokenize(text)
        if not tokens:
            return None
        ids = self._numericalize(tokens)
        lengths = torch.tensor([min(len(tokens), self.max_length)], dtype=torch.long)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = lengths.to(self.device)
        with torch.no_grad():
            logits, features = self.model(input_tensor, lengths)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        prob_dict = {label: float(probs[idx]) for idx, label in enumerate(CLASS_LABELS)}
        return {
            "probabilities": prob_dict,
            "embedding": features.squeeze(0).cpu().tolist(),
            "logits": logits.squeeze(0).cpu().tolist(),
        }
