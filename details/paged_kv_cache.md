# The Virtual Memory Allocation Era (PagedAttention / vLLM, ~2023–2024)

## Explanation
The **Virtual Memory Allocation Era** represents a major leap in serving throughput for LLMs, pioneered by the **PagedAttention** algorithm and the **vLLM** system. It directly addressed the memory footprint bottleneck of the Key-Value (KV) cache.

### Mechanism
Similar to how virtual memory in operating systems manages physical RAM, PagedAttention partitions the KV cache of a request into fixed-size block segments (e.g., storing 4 or 8 tokens per block).
* **Virtual Blocks**: The logical KV cache is divided into logical blocks.
* **Physical Blocks**: VRAM is pre-allocated as a pool of non-contiguous physical blocks.
* **Block Table**: A lookup table maps logical blocks to physical blocks. During self-attention computation, specialized CUDA kernels fetch keys and values block-by-block, resolving their physical addresses dynamically.

### Significance
By using non-contiguous memory, PagedAttention eliminated the need to allocate sequential VRAM blocks, allowing systems to dynamically allocate blocks as tokens are generated.

### Advantages
* **Near-Zero Waste**: VRAM waste due to memory fragmentation dropped from over 60% down to under 4%, freeing up massive headroom.
* **Higher Concurrency**: Allowed servers to scale their batch sizes significantly (often 2x to 4x higher throughput), dramatically reducing per-token serving costs.
* **Efficient Sharing**: Facilitated efficient KV cache sharing for parallel sampling and beam search via copy-on-write mappings.

### Limitations
* **Kernel Complexity**: Requires highly custom, complex GPU CUDA or Triton kernels to perform block-based scattered reads.
* **Overhead**: Introducing block lookups adds a minor scheduling overhead.

---

## Architecture Diagram
```mermaid
flowchart TD
    Logical[Logical KV Cache: Block 0 | Block 1 | Block 2]
    Logical --> BlockTable{"Block Table (Map)"}
    
    subgraph Physical VRAM Pool
        BlockTable --> Block0[Physical Block 12: Tokens 0-3]
        BlockTable --> Block1[Physical Block 45: Tokens 4-7]
        BlockTable --> Block2[Physical Block 7: Tokens 8-11]
    end
```

---

[Back to README](../README.md)
