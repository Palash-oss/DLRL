import random
import collections
from pathlib import Path
from typing import Deque, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        gamma: float = 0.95,
        lr: float = 1e-4,
        epsilon_start: float = 0.4,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 4000,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "backend/checkpoints",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._load_weights()

    def _epsilon(self) -> float:
        ratio = min(self.steps_done / max(self.epsilon_decay, 1), 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def select_action(self, state: torch.Tensor) -> int:
        self.steps_done += 1
        if random.random() < self._epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_values = self.policy_net(state.to(self.device))
            return int(torch.argmax(q_values).item())

    def push_transition(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def optimize(self, batch_size: int = 32) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        transitions = self.buffer.sample(batch_size)
        state_batch = torch.tensor(transitions.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(transitions.next_state, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(transitions.done, dtype=torch.bool, device=self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_q = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_q[~done_batch] = self.target_net(next_state_batch[~done_batch]).max(1)[0]
        expected_q = reward_batch + self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self) -> None:
        torch.save(self.policy_net.state_dict(), self.checkpoint_dir / "dqn_policy.pt")
        torch.save(self.target_net.state_dict(), self.checkpoint_dir / "dqn_target.pt")

    def _load_weights(self) -> None:
        policy_path = self.checkpoint_dir / "dqn_policy.pt"
        target_path = self.checkpoint_dir / "dqn_target.pt"
        if policy_path.exists() and target_path.exists():
            try:
                self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
                self.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
                print("✓ DQN checkpoints loaded")
            except Exception as exc:
                print(f"Warning: could not load DQN checkpoints ({exc}); starting fresh.")


class CorrectionLayer(nn.Module):
    def __init__(self, state_dim: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def apply_action_bias(probabilities: torch.Tensor, action_idx: int, delta: float = 0.1) -> torch.Tensor:
    probs = probabilities.clone()
    if action_idx == 1:
        probs[2] += delta
    elif action_idx == 2:
        probs[1] += delta
    elif action_idx == 3:
        probs[0] += delta
    probs = torch.clamp(probs, 1e-6, 1.0)
    return probs / probs.sum()
