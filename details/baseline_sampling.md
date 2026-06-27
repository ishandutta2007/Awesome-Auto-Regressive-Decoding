# The Baseline Sampling Era (Pre-2023)

## Explanation
The **Baseline Sampling Era** refers to the foundational period in Large Language Model (LLM) autoregressive decoding prior to the adoption of advanced cache management and parallelized verification architectures. In this era, sequence generation was performed using standard sequential, token-by-token heuristics.

### Mechanism
In a baseline autoregressive decoder:
1. The model takes a prompt sequence of length $N$.
2. It processes the prompt in a pre-fill step to generate the first token $T_{N+1}$.
3. For each subsequent token $T_t$ (where $t > N+1$), the model must run a full forward pass. 
4. Crucially, without modern block-level virtual memory caching, systems either had to re-compute Key-Value (KV) attention states for all historical tokens at every single step, or allocate large, fixed-size contiguous memory blocks in GPU VRAM to store the accumulating KV cache.

```
Prompt tokens -> [ GPU Forward Pass ] -> Token 1
[Prompt + Token 1] -> [ GPU Forward Pass ] -> Token 2
[Prompt + Token 1 + Token 2] -> [ GPU Forward Pass ] -> Token 3
```

### Significance
It established the primary functional decoding strategies—such as Greedy Search, Top-k, and Top-p (Nucleus) sampling—that are still mathematically applied today. However, it exposed severe memory bandwidth limitations when scaling to production workloads.

### Advantages
* **Simplicity**: Extremely simple to implement in standard deep learning frameworks (e.g., PyTorch, Hugging Face Transformers) without specialized custom CUDA kernels.
* **Deterministic Baseline**: Provides a straightforward implementation that serves as the correctness reference for all optimization techniques.

### Limitations
* **Severe Memory Waste**: Contiguous allocation led to internal and external VRAM fragmentation (often wasting up to 60-80% of memory).
* **High Bandwidth Memory (HBM) Stalls**: The GPU was repeatedly stalled, reading the entire model parameters and KV cache from HBM to SRAM for every single token generated.
* **Poor Scalability**: Made concurrent multi-user serving on a single GPU highly inefficient due to rapid Out-of-Memory (OOM) triggers.

---

## Architecture Diagram
```mermaid
flowchart TD
    Prompt[Prompt Input] --> PreFill["Pre-fill Step: GPU Processes Prompt"]
    PreFill --> Token1["Generate Token 1"]
    
    subgraph Iterative Loop (Re-read/Re-compute)
        Token1 --> ForwardPass["Read Entire KV Cache + Model Weights from VRAM"]
        ForwardPass --> GenToken["Generate Token t"]
        GenToken --> Append["Append Token t to History (Static VRAM Allocation)"]
        Append --> ForwardPass
    end
```

---

[Back to README](../README.md)
