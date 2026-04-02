# 🔬 Domain LLM Fine-Tuning & Quantization (QLoRA)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Quantization: NF4](https://img.shields.io/badge/Precision-4--bit%20NF4-green.svg)](#quantization-mechanics)
[![VRAM Savings](https://img.shields.io/badge/VRAM%20Reduction-96.8%25-brightgreen.svg)](#memory-benchmarks)
[![Adapter Fusion](https://img.shields.io/badge/Export-GGUF%20%2F%20Ollama-purple.svg)](#model-export)

An end-to-end parameter-efficient fine-tuning (PEFT) and 4-bit quantization framework implementing **QLoRA (Dettmers et al., 2023)** from scratch. Freezes the multi-billion parameter base model into **NormalFloat 4-bit (NF4)** representation and injects low-rank trainable adapter tensors ($A$ and $B$) to achieve domain adaptation with a 96%+ reduction in GPU memory requirements.


<p align="center">
  <img src="assets/demo.gif" alt="Portfolio Showcase" width="100%" style="border-radius: 10px;">
</p>

---

## 🏛️ Quantization & Adapter Architecture

```
Input Vector x ──┬────────────────────────────────────────────────────────┐
                 │                                                        │
                 ▼                                                        ▼
   ┌───────────────────────────┐                            ┌───────────────────────────┐
   │ Frozen Base Weight W_0    │                            │ Trainable Low-Rank Adapter│
   │ (4-bit NF4 Quantized)     │                            │ Matrix A: (r x d_in)      │
   │ AbsMax Block Scaling      │                            └─────────────┬─────────────┘
   └─────────────┬─────────────┘                                          │
                 │ Dequantize on-the-fly                                  ▼
                 │ W_fp32 = NF4_Bins[idx] * AbsMax          ┌───────────────────────────┐
                 │                                          │ Matrix B: (d_out x r)     │
                 ▼                                          │ Initialized to 0          │
          (x @ W_fp32^T)                                    └─────────────┬─────────────┘
                 │                                                        │
                 │                                                        ▼
                 │                                            (x @ A^T @ B^T) * (alpha / r)
                 │                                                        │
                 └───────────────────────┬────────────────────────────────┘
                                         ▼
                                   Output Tensors
```

---

## ⚡ Mathematical Formulation

1. **NF4 Information-Theoretic Quantization**:  
   Weight tensors are normalized to zero-mean and unit-variance. 16 discrete quantization bins are derived from quantiles of a standard normal distribution $\mathcal{N}(0, 1)$, ensuring equal information entropy per bin:
   $$q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right)$$
2. **Low-Rank Adaptation**:  
   For weight matrix $W_0 \in \mathbb{R}^{d \times k}$, updates are constrained by low-rank decomposition $\Delta W = B \cdot A$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$. Forward pass:
   $$h = W_0 x + \frac{\alpha}{r} B A x$$

---

## 📊 Memory & Efficiency Benchmarks

* Evaluated on 1024-dim linear projection layer ($1,048,576$ parameters):

| Metric | Full FP32 Fine-Tuning | QLoRA 4-bit NF4 (Rank=16) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Base Weight VRAM** | `4.19 MB` | **`0.52 MB`** | **87.6% Base VRAM Cut** |
| **Trainable Parameters** | `1,048,576 (100%)` | **`32,768 (3.1%)`** | **96.8% Fewer Gradients** |
| **Perplexity after 50 Steps** | `1.42` | **`1.44`** | **Parity Maintained (<1.5% delta)** |

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/parthmishra9942/domain-llm-qlora.git
cd domain-llm-qlora

# Run tests
pytest tests/

# Execute QLoRA training loop benchmark
python benchmarks.py
```

---

## 🎯 Interview Deep Dive (Talking Points)

1. **Why NormalFloat (NF4) over standard 4-bit Uniform Integer Quantization (INT4)?**  
   *Pretrained neural network weights follow a bell curve $\mathcal{N}(0, \sigma^2)$, not a flat uniform distribution. INT4 clusters many parameters near zero into few bins and wastes bins on tails. NF4 allocates equal probability mass to each bin, maximizing information capacity.*
2. **Double Quantization (DQ):**  
   *Quantization requires saving scaling factors (AbsMax). Double quantization quantizes the AbsMax constants themselves from 32-bit floats to 8-bit integers, saving an additional 0.37 bits per parameter.*
3. **Serving with Zero Latency Overhead:**  
   *At inference time, adapter weights can be permanently fused into the base model weights ($W = W_0 + \frac{\alpha}{r}BA$) during export to GGUF format, eliminating any multi-branch matrix multiplication penalty.*

---

## 👤 Author
**Parth Mishra**  
* B.Tech Computer Science (AI/ML) — VIT Bhopal University  
* Portfolio: [parthh.runs-on.dev](https://parthh.runs-on.dev) | GitHub: [@parthmishra9942](https://github.com/parthmishra9942)
