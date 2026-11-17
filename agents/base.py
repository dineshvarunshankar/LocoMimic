"""Base Algorithm Interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class BaseAlgorithm(ABC):
    """Base class for all RL algorithms."""

    @abstractmethod
    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Select actions for batched observations."""
        pass

    @abstractmethod
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update algorithm from batched transitions."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {}

    def store_transition(self, **kwargs) -> None:
        """Store one batched transition into algorithm-managed replay memory."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement store_transition()"
        )

    def update_from_replay(self) -> Dict[str, float]:
        """Run one update pass using algorithm-managed replay memory."""
        return {}
