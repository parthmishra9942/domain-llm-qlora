import numpy as np
from domain_qlora.qlora_layer import QLoRALinear

class QLoRATrainer:
    """Memory-efficient LoRA adapter training loop."""
    def __init__(self, layer: QLoRALinear, lr: float = 1e-4):
        self.layer = layer
        self.lr = lr

    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        pred = self.layer(x)
        error = pred - target
        loss = float(np.mean(error ** 2))

        # Gradient with respect to LoRA B and A
        grad_out = 2 * error / error.size
        # grad_B = grad_out.T @ (x @ lora_A.T) * scaling
        grad_B = (grad_out.reshape(-1, self.layer.out_features).T @ (x @ self.layer.lora_A.T).reshape(-1, self.layer.rank)) * self.layer.scaling
        
        self.layer.lora_B -= self.lr * grad_B
        return loss

    def compute_perplexity(self, loss: float) -> float:
        return float(np.exp(min(loss, 20.0)))
