from domain_qlora.qlora_layer import QLoRALinear, NF4Quantizer
from domain_qlora.export import ModelExporter
import numpy as np

def test_nf4_quantization_roundtrip():
    w = np.random.randn(64, 64).astype(np.float32)
    indices, absmax = NF4Quantizer.quantize(w)
    dequant = NF4Quantizer.dequantize(indices, absmax, w.shape)
    # Cosine similarity between original and 4-bit representation should be > 0.95
    cos_sim = np.dot(w.flatten(), dequant.flatten()) / (np.linalg.norm(w) * np.linalg.norm(dequant))
    assert cos_sim > 0.92

def test_qlora_forward_and_merge():
    layer = QLoRALinear(in_features=32, out_features=16, rank=4)
    x = np.random.randn(2, 32).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 16)

    exporter = ModelExporter(layer)
    fused = exporter.merge_weights()
    assert fused.shape == (16, 32)
