# FlashAttention Kernel Fusion

## Explanation
**FlashAttention** is a hardware-aware attention algorithm designed to speed up transformer training and inference by optimizing GPU memory reads/writes.

### Mechanism
Standard self-attention computes the intermediate similarity matrix $S = QK^T$ and attention weights $P = \text{softmax}(S)$ in High Bandwidth Memory (HBM). This requires $O(N^2)$ memory reads and writes, which bottleneck modern GPUs.
FlashAttention modifies this calculation:
1. **Tiling**: Divides the query, key, and value matrices into smaller blocks.
2. **On-Chip SRAM**: Loads blocks from HBM into the fast, on-chip GPU SRAM.
3. **Incremental Softmax**: Computes softmax values incrementally by tracking scaling factors across tiles, avoiding the need to write the massive $O(N^2)$ intermediate attention matrix back to HBM.
4. **Kernel Fusion**: Fuses the matrix multiplication, masking, and softmax operations into a single CUDA kernel.

### Significance
It bypassed the physical memory bandwidth bottleneck of self-attention, allowing models to scale to long context windows (e.g., 32k, 100k+ tokens).

### Advantages
* **Speedup**: 2x to 4x faster attention computation during training and inference.
* **Memory Reduction**: Drops the memory footprint from quadratic ($O(N^2)$) to linear ($O(N)$) in terms of sequence length.

### Limitations
* **Hardware Specific**: Heavily optimized for specific GPU architectures (Ampere, Hopper, etc.) and requires custom CUDA or Triton kernel implementations.

---

## Architecture Diagram
```mermaid
flowchart TD
    subgraph GPU HBM (Slow)
        Q[Query Matrix]
        K[Key Matrix]
        V[Value Matrix]
        Out[Output Matrix]
    end
    
    subgraph GPU SRAM (Fast Registers)
        Q_tile[Query Tile]
        K_tile[Key Tile]
        V_tile[Value Tile]
        
        Q --> |Load Tile| Q_tile
        K --> |Load Tile| K_tile
        V --> |Load Tile| V_tile
        
        Compute["Compute Attention + Softmax Incrementally"]
        Q_tile & K_tile & V_tile --> Compute
    end
    
    Compute --> |Write Output Tile| Out
```

---

[Back to README](../README.md)
