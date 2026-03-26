import numpy as np
import json
from domain_qlora.qlora_layer import QLoRALinear, NF4Quantizer

class ModelExporter:
    """Fuses LoRA adapter weights back into base weights and generates GGUF metadata."""
    def __init__(self, layer: QLoRALinear):
        self.layer = layer

    def merge_weights(self) -> np.ndarray:
        w_base = NF4Quantizer.dequantize(self.layer.q_indices, self.layer.absmax, self.layer.orig_shape)
        delta_w = (self.layer.lora_B @ self.layer.lora_A) * self.layer.scaling
        fused = w_base + delta_w
        return fused

    def export_gguf_metadata(self, model_name: str = "Clinical-Llama-3-8B-Q4") -> dict:
        metadata = {
            "format": "GGUF-v3",
            "model_name": model_name,
            "architecture": "llama",
            "quantization": "q4_K_M",
            "parameters": "8.03B",
            "lora_rank": self.layer.rank,
            "adapter_scaling": self.layer.scaling,
            "target_modules": ["q_proj", "v_proj"],
            "chat_template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        }
        return metadata
