# Methodology: Unified Hybrid Collaborative Reasoning

Systematic reasoning requires more than just predicting the next word; it requires maintaining a coherent internal chain of thought while remaining robust to local errors. In this work, we introduce a unified **Hybrid Reasoning and Multi-Path Collaboration** framework. Unlike standard autoregressive models that are constrained to discrete token sequences, our approach operates in a hybrid space—simultaneously traversing a discrete "action" path (tokens) and a continuous "thought" path (latent states). Furthermore, we ensure these thoughts are not isolated; they are dynamically refined through collaborative interaction with parallel reasoning peer trajectories.

## The Hybrid-Collaborative Generative Cycle

We fundamentally alter the Transformer's generation cycle. In a standard Large Language Model, the process at timestep $t$ is strictly sequential and isolated: the model computes a hidden state $\hat{h}_t$, projects it to a probability distribution $p_{t+1}$, samples a discrete token $x_{t+1}$, and feeds its embedding $e(x_{t+1})$ into the next step.

Our framework intervenes in this cycle to inject latent reasoning and peer context. We formalize this as a three-stage process occurring at every "thinking" step $t$: **Latent Projection**, **Collaborative Refinement**, and **Gated Injection**.

### 1. Latent Thought Projection (The "Thought")
First, we extract a continuous representation of the model's uncertainty and intent. When the model computes the probability distribution $p_{t+1}$ over the vocabulary $V$, it effectively defines a "cloud" of potential next directions. Rather than collapsing this immediately into a single discrete token, we project this entire distribution back into the semantic embedding manifold. We compute a **raw latent thought** $\tilde{h}_{t+1}$ as the probability-weighted centroid of the vocabulary embeddings $W_e$:
$$ \tilde{h}_{t+1} = \frac{\sum_{v \in V} p_{t+1}[v] \cdot W_e[v]}{||\sum_{v \in V} p_{t+1}[v] \cdot W_e[v]||_2} $$
By normalizing this vector, we ensure it lies on the same hypersphere as standard token embeddings, preserving the scale and variance expected by the transformer's internal layers. This $\tilde{h}_{t+1}$ represents the model's "expected" next move in continuous space, capturing nuances lost in discrete sampling.

### 2. Collaborative Refinement (The "Peer Review")
In isolation, a model might be confident in an incorrect reasoning step. To mitigate this, we generate $N$ parallel rollouts $\{\tau_1, ..., \tau_N\}$ simultaneously. Before committing to their thoughts, these paths interact to refine their latent states.

For each path $i$, we look at its peers to find logically consistent allies. We calculate a similarity matrix $S^{(t)}$ based on the alignment of their probability distributions $p^{(t)}$. Paths that are "thinking" similar things (high cosine similarity) support each other:
$$ S_{ij}^{(t)} = \text{CosineSimilarity}(p_i^{(t)}, p_j^{(t)}) $$

Using a temperature-scaled attention mechanism, path $i$ aggregates a **peer context vector** $c_i^{(t)}$ from the latent thoughts of its neighbors $\tilde{h}_j^{(t)}$. This context allows the model to "peek" at alternative valid reasoning patterns. We then fuse this global context into the local path to produce a **refined thought** $h_{i, t+1}^*$:
$$ h_{i, t+1}^* = (1 - \lambda)\tilde{h}_{i, t+1} + \lambda \sum_{j \neq i} \text{Softmax}\left(\frac{S_{ij}^{(t)}}{T}\right) \tilde{h}_{j, t+1} $$
Here, $\lambda$ controls the blending strength. If $\lambda=0$, the model relies solely on its own internal logic; as $\lambda$ increases, it incorporates more consensus from the group, effectively suppressing outlier hallucinations.

### 3. Gated Injection (The "Action")
Finally, the model must take a concrete step to advance the sequence. We sample a discrete token $x_{t+1} \sim p_{t+1}$ as the specific "action" for this step. However, feeding only $e(x_{t+1})$ into the next step would discard the rich, refined continuous information we just computed.

To preserve both the precision of the discrete token and the richness of the refined thought, we employ a **Thinking Residual Gate**. This learnable module effectively decides how much of the next input should be the specific word $x_{t+1}$ and how much should be the abstract thought $h_{t+1}^*$. The input to the transformer at step $t+1$ becomes:
$$ e_{t+1}^{\text{input}} = \text{Gate}\left(e(x_{t+1}), h_{t+1}^*\right) $$
where $\text{Gate}(\cdot)$ is a GRU-like mechanism that updates the blending dynamically based on the current context. This allows the model to be "vague" in its internal representation when the reasoning path is uncertain, or "precise" when the flow is clear.

## Training via Group Relative Policy Optimization

We optimize this entire hybrid-collaborative pipeline end-to-end using Reinforcement Learning. Since we cannot obtain ground-truth labels for "thoughts," we rely on the correctness of the final answer to supervise the process.

Using **Group Relative Policy Optimization (GRPO)**, we estimate the advantage of each trajectory $A(\tau_i)$ by comparing its reward $R(\tau_i)$ against the group average $\mu_R$. The gradients backpropagate through time, updating not only the language model's weights to improve $p_{t+1}$ but also the parameters of the **Thinking Residual Gate**. This teaches the model not just *what* to say, but *how* to balance its reliance on discrete tokens versus collaborative latent thoughts to maximize final reasoning accuracy.
