# The Key-Value (KV) Cache Memory Wall

## Explanation
**The Key-Value (KV) Cache Memory Wall** is one of the most critical engineering bottlenecks in scaling LLM sequence lengths and serving concurrency.

### Mechanism
During autoregressive decoding, the model requires the Key ($K$) and Value ($V$) attention projection vectors of all past tokens to compute self-attention for the current token. Storing these vectors in GPU memory prevents redundant computations.
However, this cache grows linearly with context length ($N$), batch size ($B$), number of attention heads ($H$), hidden dimension size ($D$), and network layer depth ($L$):
$$\text{Memory}_{KV} = 2 \times B \times L \times H \times D \times N \times \text{Bytes-per-parameter}$$
For a Llama-3-70B model serving a batch size of 32 at a 32k context length, the KV cache requires dozens of gigabytes of VRAM—often exceeding the memory footprint of the model weights themselves.

### Significance
This memory bottleneck restricts serving capacity, limiting maximum batch sizes and context lengths, and often causing Out-of-Memory (OOM) errors during long-context serving.

### Advantages of Mitigations
Implementing solutions like Grouped-Query Attention (GQA) and sliding-window attention (e.g., StreamingLLM) significantly reduces memory requirements:
* **Grouped-Query Attention (GQA)**: Compresses KV cache sizes by mapping multiple query heads to a single key/value head (typically reducing KV cache size by 8x).
* **StreamingLLM (Attention Sinks)**: Keeps only the first few tokens (attention sinks) and a rolling window of recent tokens, enabling infinite generation with a fixed memory footprint.

### Limitations
* **GQA**: Requires architectural support and model retraining.
* **StreamingLLM**: Drops intermediate context history, which can degrade retrieval performance in tasks requiring long-range memory.

---

## Architecture Diagram
```mermaid
flowchart TD
    subgraph Multi-Head Attention (Standard)
        Q_Heads[8 Query Heads] <--> KV_Heads_MHA[8 Key-Value Heads]
        note1["KV Cache Size: 100%"]
    end
    
    subgraph Grouped-Query Attention (Mitigation)
        Q_Heads_GQA[8 Query Heads] <--> KV_Heads_GQA[2 Key-Value Heads]
        note2["KV Cache Size: 25% (Compressed)"]
    end
    
    subgraph StreamingLLM (Mitigation)
        Sinks[First 4 Tokens - Sinks] --> Attention
        Window[Sliding Window of Recent Tokens] --> Attention
        note3["Memory: Constant O(1)"]
    end
```

---

[Back to README](../README.md)
