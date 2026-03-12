import numpy as np

class NF4Quantizer:
    """NormalFloat 4-bit (NF4) Quantization (Dettmers et al., 2023)."""
    # 16 quantization bins optimized for standard normal distribution
    NF4_BINS = np.array([
        -1.0, -0.6961928, -0.5250730, -0.3949175,
        -0.2844414, -0.1845100, -0.0910500, 0.0,
        0.0795802, 0.1609302, 0.2461123, 0.3379152,
        0.4407098, 0.5626170, 0.7229568, 1.0
    ], dtype=np.float32)

    @classmethod
    def quantize(cls, weight: np.ndarray, block_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        flat = weight.flatten()
        pad_len = (block_size - (len(flat) % block_size)) % block_size
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len))
        blocks = flat.reshape(-1, block_size)
        absmax = np.max(np.abs(blocks), axis=1, keepdims=True)
        absmax[absmax == 0] = 1.0
        normalized = blocks / absmax

        # Map to nearest NF4 bin indices (0..15)
        indices = np.abs(normalized[:, :, np.newaxis] - cls.NF4_BINS).argmin(axis=-1)
        return indices.astype(np.uint8), absmax.astype(np.float32)

    @classmethod
    def dequantize(cls, indices: np.ndarray, absmax: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        dequant = cls.NF4_BINS[indices] * absmax
        flat = dequant.flatten()
        return flat[:np.prod(orig_shape)].reshape(orig_shape)

class QLoRALinear:
    """Quantized 4-bit Base Layer with Low-Rank Adapters (LoRA)."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen 4-bit quantized base weight
        raw_w = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.q_indices, self.absmax = NF4Quantizer.quantize(raw_w)
        self.orig_shape = (out_features, in_features)

        # Trainable low-rank adapters: B initialized to 0, A to Gaussian
        self.lora_A = np.random.randn(rank, in_features).astype(np.float32) * (1.0 / np.sqrt(rank))
        self.lora_B = np.zeros((out_features, rank), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Base model forward: W_dequant @ x.T
        w_base = NF4Quantizer.dequantize(self.q_indices, self.absmax, self.orig_shape)
        base_out = x @ w_base.T

        # LoRA adapter path: (x @ A.T @ B.T) * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out
