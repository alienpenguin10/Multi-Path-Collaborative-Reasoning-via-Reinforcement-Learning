# Methodology

## Contributions

In this work, we propose a novel framework for enhancing the reasoning capabilities of Large Language Models (LLMs) through latent space exploration and collaborative generation. Our core contributions are as follows:

1.  **Hybrid Latent Reasoning:** We introduce a mechanism that transitions from discrete token generation to continuous latent "thoughts" during reasoning phases. By projecting hidden states back into the embedding manifold and blending them via a parameter-free gating mechanism, we enable the model to maintain semantic ambiguity and unexplored reasoning paths without committing prematurely to discrete tokens.
2.  **Multi-Path Collaborative Reasoning:** We propose a cross-path interaction module where multiple parallel rollouts ($N$) dynamically share information. This allows individual trajectories to leverage insights from peer generations through a probability-distribution-based similarity attention mechanism, fostering global consistency across the reasoning group.
3.  **Reinforcement Learning Optimization:** We effectively train these mechanisms using Group Relative Policy Optimization (GRPO). Unlike traditional Chain-of-Thought (CoT) distillation, our approach optimizes the latent gating parameters end-to-end based on the final answer correctness, remaining largely parameter-efficient while fundamentally shifting the reasoning paradigm.

## Preliminaries

We begin by formally defining the standard autoregressive generation process in Transformer-based LLMs. Let $x = [x_1, x_2, ..., x_t]$ denote the input query sequence. These discrete tokens are mapped to continuous representations via an embedding matrix $W_e \in \mathbb{R}^{|V| \times d}$, resulting in token embeddings $E = [e_1, e_2, ..., e_t]$. To incorporate sequence order, positional encodings $P$ are added to form the position-aware embeddings $E' = E + P$.

The input embeddings $E'$ are processed by a Transformer encoder, consisting of stacked layers of multi-head self-attention (MHSA) and feed-forward networks (FFN), each followed by residual connections and layer normalization. The final hidden state at step $t$, denoted as $\hat{h}_t \in \mathbb{R}^d$, encapsulates the contextual representation of the sequence up to time $t$:
$$ \hat{H} = \text{Transformer}(E') = [\hat{h}_1, \hat{h}_2, ..., \hat{h}_t] $$

To generate the next token, the final hidden state $\hat{h}_t$ is projected by the language modeling head (`Head`) to produce logits over the vocabulary $V$. The probability distribution for the next token $x_{t+1}$ is obtained via the softmax function, often modulated by a temperature parameter $\tau$:
$$ p_{t+1} = \text{Softmax}\left(\frac{\text{Head}(\hat{h}_t)}{\tau}\right) $$

In standard autoregressive decoding, a discrete token $\hat{x}_{t+1}$ is sampled from this distribution, $\hat{x}_{t+1} \sim p_{t+1}$, and its embedding $e(\hat{x}_{t+1})$ is fed back into the model for the subsequent step.

## Hybrid Latent Reasoning (HRPO)

While effective for general text generation, the discrete nature of standard decoding forces the model to collapse the probability distribution into a single choice at every step. For complex reasoning tasks, this premature discretization can discard valuable alternatives. To address this, we propose **Hybrid Reasoning Policy Optimization (HRPO)**, which allows the model to "think" in a continuous latent space.

### Latent State Projection

Instead of feeding the embedding of a single sampled token during the reasoning phase ($t \in \mathcal{T}_{\text{think}}$), we compute a "hybrid" input embedding. This embedding represents the expected semantic value of the next step by taking a probability-weighted average over the entire vocabulary's embedding space.

Simply feeding the raw hidden state $\hat{h}_t$ is insufficient, as it lies outside the manifold of valid token embeddings, leading to distribution shift and generation degradation. To align the latent thought with the model's native input space, we project the probability distribution $p_{t+1}$ back onto the embedding hypersphere:
$$ \tilde{h}_{t+1} = \frac{\sum_{v \in V} p_{t+1}[v] \cdot W_e[v]}{||\sum_{v \in V} p_{t+1}[v] \cdot W_e[v]||_2} $$
This formulation ensures that $\tilde{h}_{t+1}$ is a valid vector within the embedding space, preserving the scale and variance expected by the transformer layers while capturing the full uncertainty of the model's prediction.

### Gated Injection Mechanism

To seamlessly integrate this continuous thought into the generation stream, we employ a learnable gating mechanism inspired by Gated Recurrent Units (GRUs). This mechanism dynamically governs the mixing between the discrete sampled token $e(\hat{x}_{t+1})$ and the continuous latent thought $\tilde{h}_{t+1}$. The update rules are defined as follows:

$$
\begin{aligned}
r_t &= \sigma(W_r \hat{x}_{t+1} + b_r) \\
z_t &= \sigma(W_z \hat{x}_{t+1} + b_z) \\
n_t &= \tanh(W_n [\hat{x}_{t+1} \circ r_t] + b_n) \\
e_{t+1}^{\text{hybrid}} &= (1 - z_t) \circ n_t + z_t \circ \tilde{h}_{t+1}
\end{aligned}
$$

Note that in our implementation, we simplify this to a "Thinking Residual" connection where the gate learns to blend the standard embedding with the projected latent state, effectively allowing the model to decide per-step whether to rely on concrete textual grounding or abstract latent reasoning.

## Multi-Path Collaborative Reasoning (M3PO)

While HRPO improves individual reasoning chains, it does not prevent the model from pursuing logical fallacies in isolation. To enforce global consistency and robustness, we introduce **Multi-Path Perception Policy Optimization (M3PO)**.

In this framework, the policy $\pi_{\theta}$ generates $N$ parallel rollouts $\{\tau_1, \tau_2, ..., \tau_N\}$ for a given input query $x$. During the thinking phase, these paths do not evolve independently; they interact via a parameter-free collaborative mechanism.

### Cross-Path Contextual Attention

At each thinking step $l$, we assess the logical similarity between any two paths $i$ and $j$ by comparing their output probability distributions $p_i^{(l)}$ and $p_j^{(l)}$. We utilize the cosine similarity between distributions:
$$ S_{ij}^{(l)} = \frac{p_i^{(l)} \cdot p_j^{(l)}}{||p_i^{(l)}|| \cdot ||p_j^{(l)}||} $$

Based on this similarity, we compute cooperative attention weights $A_{ij}^{(l)}$ to determine the influence of path $j$ on path $i$. We apply a temperature-scaled softmax operation, masking out self-interactions and inactive paths:
$$ A_{ij}^{(l)} = \text{Softmax}\left(\frac{S_{ij}^{(l)}}{T_{\text{cross}}}\right) $$

### Collaborative Blending

Each trajectory $i$ then aggregates a "peer context" vector $c_i^{(l)}$, which is a weighted sum of the active, similar peers' latent states:
$$ c_i^{(l)} = \sum_{j \neq i} A_{ij}^{(l)} \tilde{h}_j^{(l)} $$

Finally, we update the current path's latent state by fusing it with the peer context via a convex combination controlled by a hyperparameter $\lambda$:
$$ h_{i}^{(l)} \leftarrow (1 - \lambda)\tilde{h}_i^{(l)} + \lambda c_i^{(l)} $$

This mechanism ensures that trajectories maintaining logically consistent distributions reinforce each other, while outlier paths are naturally corrected or suppressed.

## Training Optimization

The entire framework is trained using **Group Relative Policy Optimization (GRPO)**. We define the objective to maximize the expected reward of the generated trajectories while maintaining stability via KL-regularization.

Given a set of $N$ rollouts and their corresponding rewards $R(\tau_1), ..., R(\tau_N)$ (based on the correctness of the final answer), we compute the advantage $A(\tau_i)$ for each trajectory using group-relative standardization:
$$ A(\tau_i) = \frac{R(\tau_i) - \mu_R}{\sigma_R} $$
where $\mu_R$ and $\sigma_R$ are the mean and standard deviation of rewards within the group.

The final policy gradient loss is formulated as:
$$ \nabla_{\theta} \mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{L} \nabla_{\theta} \log \pi_{\theta}(\tau_i^{(t)} | x, \tau_i^{<t}) \cdot A(\tau_i) \right) \right] - \beta \nabla_{\theta} D_{\text{KL}}[\pi_{\theta} \| \pi_{\text{ref}}] $$

Crucially, the gradients flow not only through the language model parameters but also through the HRPO gating modules (`thinking_residual`), effectively learning the optimal strategy for utilizing latent and collaborative thoughts to achieve correct solutions.
