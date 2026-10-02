## 1. End-to-end pipeline

```mermaid
flowchart TD
    P1["Calibrate train-only p50/p75 budgets and p99 envelope"]
    P2["Freeze 800 clean-correct tests per victim/class"]
    P3["Infer forward padding and timing capability"]
    P4["Apply per-flow hard bounds and mode"]
    P5["One chosen optimizer<br/>Hybrid / Prim-PGD / Prim-C&W"]
    P6["Realize 79 features through φ"]
    P7["Check in-search hit and source-conditioned hybrid_valid"]
    P8["Independent source-conditioned post-attack validation and ASR"]

    P1 --> P4
    P2 --> P3 --> P4
    P2 --> P5
    P4 --> P5
    P5 -->|p, D, s| P6 --> P7
    P7 -->|next candidate| P5
    P7 -->|best realized flow| P8
    P2 -->|source flow| P8
```

Train-only bounds include a DoS/DDoS rate floor. Frozen FINAL flows undergo capability-aware search; independent post-attack checks report nested raw, valid, primitive-feasible, and SP-ASR.

## 2. RealizedSearch

```mermaid
flowchart TD
    R1["Score identity flow at cost one"]
    R2["Normalize q within unit cube"]
    R3["Choose candidate proposal path"]
    R4["Surrogate φ<br/>quantize=False"]
    R5["Round p/D, clamp s, quantize φ"]
    R6["Check victim hit and hybrid_valid against source"]
    R7["Prioritize hits, cost, then margin"]
    Q{"Can afford another step?<br/>256-forward cap"}
    R8["Return best realized incumbent"]

    R1 --> R2 --> R3
    R3 -->|Gradient-based refinement| R4 --> R5
    R3 -->|Hybrid integer padding without surrogate| R5
    R5 --> R6 --> R7 --> Q
    Q -->|budget left| R3
    Q -->|stop| R8
```

Hybrid can enumerate integer padding without surrogate gradients. Free-coordinate floors support straight-through gradients; surrogate and realized forwards count toward 256. Failures rank by margin, and only realized candidates become incumbents.
