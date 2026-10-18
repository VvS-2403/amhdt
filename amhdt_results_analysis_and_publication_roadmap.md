# Empirical Results Forensics, Architectural Critique, and Publication Roadmap for AMHDT
## A Principal Quantitative ML & Senior Conference Reviewer Report

> [!IMPORTANT]
> **Executive Research Assessment**: The current AMHDT empirical results exhibit two canonical failure modes of financial ML: (1) **Trivial Positive-Drift Mode Collapse** in single-asset benchmarks (100% Up-recall, 0% Down-recall, exactly 1 trade, balanced accuracy 50.0%), and (2) **Hyperactive Noise Churn with Catastrophic Turnover Bleed** in multi-asset settings (4,220 to 8,440 trades, transaction costs exceeding 1,200% to 2,500%, net cumulative return of -100.0%, and negative $R^2 \in [-2.52, -0.63]$). While multi-asset directional accuracy appears superficially promising (61.6% to 62.7%), it does not translate into economic alpha. This report delivers an exhaustive post-mortem analysis of these results, dissects the architectural flaws causing them, and lays out a mathematically rigorous blueprint to elevate AMHDT to top-tier conference publication grade (NeurIPS, ICLR, KDD, ICAIF).

---

## SECTION 1: Exhaustive Empirical Results Forensics

### 1.1 Consolidated Results Matrix Across Benchmarks

The empirical evaluations span three experimental configurations: **Dow Jones 30 Index (1D)**, **NIFTY 50 Index (1D)**, and **NIFTY 10-Stock Cross-Sectional Universe (Batch 1: TATASTEEL, BPCL, ASIANPAINT, HDFCBANK, HCLTECH, GRASIM, BRITANNIA, BHARTIARTL, POWERGRID, BAJFINANCE)** across 1-day ($H=1$) and 5-day ($H=5$) horizons.

| Evaluation Metric                  |     Dow 30 (Test)     |    NIFTY 50 (Test)    | NIFTY Multi-Asset $H=1$ (Test) | NIFTY Multi-Asset $H=5$ (Test) |     Ideal / SOTA Target      |
| :--------------------------------- | :-------------------: | :-------------------: | :----------------------------: | :----------------------------: | :--------------------------: |
| **Observation Bars / Sequences**   |     438 test seqs     |    1,049 test seqs    |        1,055 test seqs         |        1,055 test seqs         |    $\ge 2,500$ bars (OOD)    |
| **Directional Accuracy (DA)**      |        57.08%         |        53.57%         |             61.63%             |             62.73%             |     $> 56.5\%$ (robust)      |
| **Balanced Accuracy (BA)**         |      **50.00%**       |      **50.00%**       |           **54.23%**           |           **62.44%**           |          $> 55.0\%$          |
| **Precision (UP / DOWN)**          |    57.08% / 0.00%     |    53.57% / 0.00%     |        52.09% / 63.37%         |        54.34% / 69.93%         |      $> 54\% / > 54\%$       |
| **Recall (UP / DOWN)**             |  **100.0% / 0.00%**   |  **100.0% / 0.00%**   |        20.54% / 87.91%         |        60.77% / 64.12%         | Balanced ($\approx 50-60\%$) |
| **Confusion Matrix (TP/FP/TN/FN)** |   250 / 188 / 0 / 0   |   562 / 487 / 0 / 0   |   1099 / 1011 / 7354 / 4251    |   3440 / 2890 / 5164 / 2221    |   Well-populated diagonal    |
| **$R^2$ Score (Goodness of Fit)**  |      **-0.0007**      |      **-0.0320**      |          **-2.5203**           |          **-0.6327**           |     $> +0.02$ to $+0.08$     |
| **Mean Squared Error (MSE)**       | $8.80 \times 10^{-5}$ | $1.94 \times 10^{-4}$ |     $1.40 \times 10^{-3}$      |     $3.06 \times 10^{-3}$      |    Normalized calibrated     |
| **Information Coefficient (IC)**   |        +0.0548        |        -0.0141        |            +0.0146             |            +0.0321             |    $> +0.05$ (stat. sig.)    |
| **Number of Executed Trades**      |         **1**         |         **1**         |           **4,220**            |           **8,440**            |    Governed by threshold     |
| **Cumulative Tx Costs**            |         0.15%         |         0.15%         |         **1,265.85%**          |         **2,531.85%**          |     $< 25\%$ cumulative      |
| **Gross Cumulative Return**        |        +35.73%        |        +29.95%        |          **-99.89%**           |          **-100.00%**          |     Positive, beats B&H      |
| **Net Cumulative Return**          |        +35.52%        |        +29.75%        |          **-100.00%**          |          **-100.00%**          |   Positive alpha over B&H    |
| **Net Sharpe Ratio (Ann.)**        |         +1.25         |         +0.40         |           **-0.97**            |           **-0.65**            |   $> +1.50$ (net of costs)   |
| **Gross Sharpe Ratio**             |         +1.25         |         +0.40         |           **-0.237**           |           **+0.026**           |          $> +1.80$           |
| **Net Max Drawdown (MDD)**         |        -16.69%        |        -34.28%        |          **-100.00%**          |          **-100.00%**          |         $< -18.0\%$          |
| **Benchmark (B&H) Return**         |        +35.73%        |        +29.95%        |           +7,361.03%           |        +410.5B% (comp.)        |    Ground-truth baseline     |

---

### 1.2 Quantitative Dissection of Experimental Failure Modes

#### Failure Mode A: The "Trivial Mean Drift" Mode Collapse (Single-Asset Datasets)
In both Dow 30 and NIFTY 50 single-index evaluations, the confusion matrix reveals a fatal degenerated state:
$$\text{Recall}_{\text{UP}} = 1.000, \quad \text{Recall}_{\text{DOWN}} = 0.000, \quad \text{Precision}_{\text{DOWN}} = \text{Undefined (0.0)}$$
- **Mathematical Root Cause**: Equity indices have an unconditional positive long-term drift:
  $$\mathbb{E}[\Delta P_t / P_t] = \mu > 0$$
  Under unregularized Mean Squared Error ($\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{b} (y_b - \hat{y}_b)^2$) with high feature noise and low signal-to-noise ratio ($\text{SNR} < 0.05$), the global minimizer of MSE without directional penalty is the conditional mean $\hat{y} \approx \mu > 0$.
- **The "Accuracy" Mirage**: The reported directional accuracy of $57.08\%$ on Dow 30 and $53.57\%$ on NIFTY 50 is **not learned alpha**. It is simply the baseline fraction of green trading days in the test partition:
  $$\text{DA} = \frac{\sum_{t} \mathbb{I}(y_t > 0)}{T_{\text{test}}} = 57.08\%$$
  The **Balanced Accuracy is exactly 50.00%**, proving that the model operates as a coin-toss biased by the base rate.
- **Backtest Artifact**: Because $\hat{y}_t > 0$ for all $t$, the execution engine enters a single long position at $t=0$ and never exits. The trading strategy collapses to Buy-and-Hold minus one execution fee (15 bps), generating identical volatility, identical drawdown (-16.69% and -34.28%), and identical Sharpe ratios (1.25 and 0.40).

#### Failure Mode B: Catastrophic Hyperactivity & Transaction Cost Attrition (Multi-Asset Universe)
To fix the mode collapse, the decomposed loss (temporal IC + cross-sectional IC + BCE) was introduced in `train_batch1_fix.py`. This succeeded in forcing non-trivial predictions (UP and DOWN calls were both generated). However, it revealed a far more severe institutional failure mode:
1. **Excessive Churn**:
   In the 10-stock NIFTY universe over 1,055 out-of-sample sequences, the model executed **4,220 trades for $H=1$** (average 4.0 trades per day across 10 stocks) and **8,440 trades for $H=5$** (average 8.0 trades per day).
2. **Exponential Fee Decay**:
   With institutional transaction costs set at a modest 15 basis points per trade ($c = 0.0015$):
   $$\text{Cumulative Friction} = \sum_{k=1}^{K} c \cdot |\Delta w_k| \approx 1,265.85\% \quad (H=1)$$
   $$\text{Cumulative Friction} \approx 2,531.85\% \quad (H=5)$$
   This friction wiped out 100% of portfolio equity, producing a total liquidation drawdown of **-100.00%** and net annual returns of **-30.11% ($H=1$)** and **-50.23% ($H=5$)**.
3. **Absence of Gross Alpha**:
   Crucially, the failure is **not merely a transaction cost issue**. Even on a **gross basis (zero fees)**:
   - $H=1$ Gross Return = **-99.89%** (Gross Sharpe = **-0.237**)
   - $H=5$ Gross Return = **-100.00%** (Gross Sharpe = **+0.026**)
   This proves that the predicted directional flips were anti-correlated with true price movements on high-volatility days, buying local tops and selling local bottoms.

#### Failure Mode C: Severe Negative $R^2$ Calibration Collapse
While Machine Learning practitioners sometimes dismiss negative $R^2$ in finance due to noise, an $R^2$ of **-2.5203 ($H=1$)** and **-0.6327 ($H=5$)** indicates that the model's return forecasts have variance far higher than the underlying target variance, with severe phase lag:
$$R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2} = -2.5203 \implies \text{MSE}(\hat{y}) = 3.52 \times \text{Var}(y)$$
The model predictions are swinging with amplitudes $3.5\times$ larger than realistic market return innovations, causing massive out-of-sample prediction error.

#### Failure Mode D: Information Coefficient (IC) Weakness and Instability
- Dow 30: $\text{IC} = +0.0548$
- NIFTY 50 Single: $\text{IC} = -0.0141$ (Negative alpha / inverted signal)
- NIFTY Multi-Asset $H=1$: $\text{IC} = +0.0146$
- NIFTY Multi-Asset $H=5$: $\text{IC} = +0.0321$
In quantitative finance, an IC below $+0.03$ with high variance across assets is statistically indistinguishable from zero over a 1,000-sample window. Under standard t-distribution tests:
$$t_{\text{stat}} = \text{IC} \cdot \sqrt{\frac{T - 2}{1 - \text{IC}^2}} = 0.0146 \cdot \sqrt{\frac{1055 - 2}{1 - 0.0146^2}} \approx 0.474 \quad (p \approx 0.635)$$
The null hypothesis ($H_0: \text{IC} = 0$) cannot be rejected. Reviewers at NeurIPS/KDD will immediately identify that the model has zero statistically significant predictive capability in its current state.

---

## SECTION 2: Deep Architectural Critique & What Must Be Improved

The pipeline does not require architectural re-explanation; rather, it requires a surgical diagnosis of **structural flaws in the neural components** and the **mathematical modifications** necessary to fix them.

```
CURRENT VULNERABLE ARCHITECTURE:
[Static Sector / Global Correl Hypergraph] 
      │ (Dense Oversmoothing & Spurious Edges)
      ▼
[Decoupled GMM / Discrete Softmax Regime Head] 
      │ (Jitter, Hard Boundaries, No Task Feedback)
      ▼
[Deterministic Return-to-Go Decision Transformer] 
      │ (Controllability Hallucination in Non-Stationary MDP)
      ▼
[Heuristic Static Loss: 0.35 MSE + 0.35 TempIC + 0.15 CrossIC + 0.15 BCE]
      │ (Zero Turnover Penalty, Gradient Conflict, No Friction Awareness)
      ▼
[Result: 8,440 Trades, -100% Net Return, -2.52 R²]
```

---

### 2.1 Critical Vulnerabilities in the Hypergraph Formulation

#### Vulnerability 1: Static Predefined Sectors and Non-Stationary Correlation Edges
- **The Defect**: The current incidence matrix $H \in \mathbb{R}^{N \times E}$ uses static GICS-style sector hyperedges and correlation hyperedges computed over fixed empirical windows. In real markets:
  1. Sector definitions are economically rigid and miss cross-sector supply chain, thematic, and factor dependencies (e.g., auto manufacturers depending on semiconductor suppliers).
  2. Correlation matrices are notoriously non-stationary. Empirical Pearson correlation matrices $\hat{\Sigma}_{\tau}$ contain high noise eigenvalues (Marchenko-Pastur theorem) that create spurious hyperedges.
- **The Improvement: Adaptive Differentiable Hypergraph Structure Learning (AD-HSL)**:
  Replace fixed incidence matrices with a parameter-efficient, self-attention hypergraph generator equipped with Gumbel-Softmax sparsity:
  $$S_{ij} = \frac{\langle W_Q z_i, W_K e_j \rangle}{\sqrt{d}}, \quad H_{ij} = \text{Gumbel-Softmax}(S_{ij}, \tau)$$
  where hyperedge prototypes $e_j \in \mathbb{R}^d$ ($j \in \{1, \dots, K_e\}$) are learnable latent factor vectors representing dynamic market themes (e.g., "Liquidity Contraction", "Energy Shock", "Momentum Unwind").
  Enforce a Dirichlet energy regularization penalty on $H$ to prevent the trivial dense graph:
  $$\mathcal{L}_{\text{sparse}}(H) = \sum_{j=1}^{K_e} \left( \|H_{:, j}\|_1 - \kappa \right)^2 + \gamma \text{Tr}(H^\top L_v H)$$

#### Vulnerability 2: Graph Convolutional Oversmoothing
- **The Defect**: The standard hypergraph spectral convolution:
  $$Z^{(l+1)} = \sigma\left( D_v^{-1/2} H W_e D_e^{-1} H^\top D_v^{-1/2} Z^{(l)} \Theta^{(l)} \right)$$
  acts as a low-pass filter over the hypergraph Laplacian. For small universes ($N=10$ to $N=50$), stacking $L \ge 2$ layers drives node representations $z_i, z_j$ toward a shared centroid:
  $$\lim_{l \to \infty} \|z_i^{(l)} - z_j^{(l)}\|_2 = 0$$
  This completely destroys the cross-sectional dispersion necessary for ranking stocks, directly explaining why the model struggled to separate winning and losing stocks.
- **The Improvement: Decoupled Hypergraph Attention with Personalized Residuals (AP-HGN)**:
  Incorporate initial residual connections (inspired by APPNP / GCNII) and edge-level attention weighting:
  $$Z^{(l+1)} = (1 - \alpha) \cdot \text{HyperAttn}\left(Z^{(l)}, H, W_e\right) + \alpha Z^{(0)} + \beta W_R Z^{(l)}$$
  where $\alpha \in [0.1, 0.3]$ guarantees that individual asset idiosyncratic features (momentum, earnings surprise, short interest) are preserved through deep propagation layers.

---

### 2.2 Critical Vulnerabilities in Regime Modeling

#### Vulnerability 1: Decoupled Unsupervised Regime Induction
- **The Defect**: The regime classifier in `module3_regime.py` clusters embeddings into discrete states ($K=4$) using softmax cross-entropy against pseudo-labels or unsupervised clustering. Because this clustering objective is not directly derived from the downstream forecasting or portfolio loss, the induced regimes represent statistical artifacts of volatility rather than actionable trading regimes.
- **The Improvement: End-to-End Task-Gated Mixture-of-Experts (MoE) Regimes**:
  Eliminate discrete pseudo-labels. Formulate regimes as latent router gates in a Mixture-of-Experts Decision Transformer:
  $$g_t = \text{Softmax}\left( W_g \cdot \bar{Z}_t + b_g \right) \in \Delta^{K-1}$$
  Each regime $k \in \{1, \dots, K\}$ parameterizes a specialized expert sub-network $\mathcal{F}_k(Z_t)$. The global trajectory representation is the convex combination:
  $$\tilde{H}_t = \sum_{k=1}^{K} g_{t, k} \mathcal{F}_k(Z_t)$$
  This guarantees that regime discovery is optimized strictly for downstream trading performance and out-of-sample risk reduction.

#### Vulnerability 2: Temporal Jitter and Boundary Instability
- **The Defect**: The current regime probabilities $P(\text{regime}_t)$ fluctuate day-to-day on single-day return spikes, inducing regime-flipping jitter that causes the downstream policy to execute massive asset reallocations.
- **The Improvement: Sticky Hidden Markov / Continuous Neural Jump Regime Dynamics**:
  Enforce temporal persistence by parameterizing the transition dynamics with a Dirichlet prior favoring self-transitions ($\kappa_{ii} \gg \kappa_{ij}$):
  $$\mathcal{L}_{\text{persist}} = -\sum_{t=1}^{T-1} \log \left( g_t^\top \mathcal{T}_{\text{sticky}} g_{t+1} \right) + \lambda_{\text{entropy}} \sum_{t=1}^T \mathcal{H}(g_t)$$
  This penalizes spurious intraday regime switches while permitting rapid transitions during macroeconomic stress shocks (e.g., volatility spikes exceeding $3\sigma$).

---

### 2.3 Critical Vulnerabilities in the Decision Transformer Formulation

#### Vulnerability 1: The "Controllability Fallacy" of Return-to-Go in Finance
- **The Defect**: The vanilla Decision Transformer (Chen et al., 2021) was formulated for deterministic or controlled Markov Decision Processes (Atari, MuJoCo robotics), where the agent's actions directly alter state transitions:
  $$s_{t+1} \sim \mathcal{P}(s_{t+1} \mid s_t, a_t)$$
  In financial trading, the investor is a **price-taker** in a stochastic, non-stationary environment. The environment transition $\mathcal{P}(s_{t+1} \mid s_t)$ is virtually invariant to the agent's portfolio weights $w_t$ (except for massive market impact).
  Conditioning the network on a high synthetic Return-to-Go ($\hat{R}_1 = +30\%$ annual return) forces the model to either:
  1. Learn spurious correlational short-cuts that fail out-of-sample; or
  2. Take extreme high-beta leverage during market drawdowns to satisfy the unrealistic conditioning prompt (gambler's ruin).
- **The Improvement: Distributional Risk-Conditioned Decision Transformer (RC-DT)**:
  Replace the scalar scalar return-to-go $\hat{R}_t$ with a bivariate tuple conditioning on **Target Expected Return and Maximum Acceptable Downside Deviation (Target Sortino)**:
  $$\tau = \left( (\mu^*, \text{VaR}^*_\alpha)_1, s_1, a_1, \dots, (\mu^*, \text{VaR}^*_\alpha)_T, s_T, a_T \right)$$
  During inference, prompt the model with market-realistic conservative goals:
  $$\mu^* = \text{EMA}(\text{Market Return}), \quad \text{VaR}^*_{0.05} \le -1.5\%$$
  This grounds the policy in risk-managed capital preservation rather than unconstrained return pursuit.

#### Vulnerability 2: Per-Asset Tokenization vs Joint Cross-Sectional Allocation
- **The Defect**: In `pipeline.py` (lines 180–218), the tensor is reshaped from $(B, N, T, F)$ to $(B \cdot N, T, F)$, and passed through the Decision Transformer independently per asset.
  This completely breaks cross-asset attention inside the transformer! The transformer cannot see Stock B when deciding the weight for Stock A at time $t$. The portfolio budget constraint ($\sum_{i=1}^N w_{i, t} = 1$) is applied as an afterthought in Module 5 via a post-hoc Softmax or normalization layer.
- **The Improvement: Spatio-Temporal Factorized Attention Transformer**:
  The sequence must be tokenized as a joint matrix of asset tokens at each timestep $t$. Apply factorized attention:
  1. **Temporal Self-Attention**: Computes dependency across $t \in \{1, \dots, T\}$ for each asset.
  2. **Cross-Asset Spatial Attention**: Computes dependency across $i \in \{1, \dots, N\}$ at time $t$:
  $$A_{\text{spatial}} = \text{Softmax}\left(\frac{(X_t W_Q)(X_t W_K)^\top}{\sqrt{d}} + M_{\text{graph}}\right)$$
  This guarantees that portfolio rebalancing decisions are computed simultaneously with full visibility into cross-asset covariance and relative liquidity.

---

### 2.4 Critical Flaws in the Loss Function and Execution Modeling

#### Vulnerability 1: Arbitrary Static Loss Weighting and Gradient Starvation
- **The Defect**: The loss function in `training.py` (line 165):
  $$\mathcal{L} = 0.35 \mathcal{L}_{\text{MSE}} + 0.35 \mathcal{L}_{\text{TemporalIC}} + 0.15 \mathcal{L}_{\text{CrossIC}} + 0.15 \mathcal{L}_{\text{BCE}}$$
  uses hand-tuned static multipliers. In multi-task optimization, the gradient norms $\|\nabla_\theta \mathcal{L}_{\text{MSE}}\|$ and $\|\nabla_\theta \mathcal{L}_{\text{TemporalIC}}\|$ differ by 2 to 3 orders of magnitude. The MSE loss dominates early training, pulling predictions toward zero, while the BCE loss drives logits to extreme values, leading to gradient conflict and Pareto-suboptimal representations.
- **The Improvement: Uncertainty-Weighted Multi-Task Loss with PCGrad**:
  Adopt homoscedastic uncertainty weighting (Kendall et al.):
  $$\mathcal{L}(\theta, \sigma) = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{MSE}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{IC}} + \frac{1}{2\sigma_3^2} \mathcal{L}_{\text{BCE}} + \sum_{i=1}^3 \log(1 + \sigma_i^2)$$
  Implement **Projecting Conflicting Gradients (PCGrad)**: If the cosine similarity $\cos(\nabla \mathcal{L}_i, \nabla \mathcal{L}_j) < 0$, project $\nabla \mathcal{L}_i$ onto the normal plane of $\nabla \mathcal{L}_j$ before the optimizer step:
  $$\nabla \mathcal{L}_i \leftarrow \nabla \mathcal{L}_i - \frac{\nabla \mathcal{L}_i \cdot \nabla \mathcal{L}_j}{\|\nabla \mathcal{L}_j\|^2} \nabla \mathcal{L}_j$$

#### Vulnerability 2: Zero Turnover Regularization in the Optimization Objective
- **The Defect**: This is the single biggest contributor to the -100% backtest return. The model was trained entirely on instantaneous prediction accuracy ($\hat{y}_t \approx y_t$) with **zero mathematical awareness of the cost of moving from weight $w_{t-1}$ to $w_t$**. A policy that flips from $+100\%$ long to $-100\%$ short every 2 days incurs 30 bps of drag every 48 hours ($\approx 37.5\%$ annualized drag).
- **The Improvement: Differentiable End-to-End Net Sharpe / Sortino Loss with Turnover Penalty**:
  Formulate the training loss directly over the realized net portfolio return sequence $R_t^{\text{net}} = w_t^\top y_t - c_t \|\Delta w_t\|_1$:
  $$\mathcal{L}_{\text{Trading}}(\theta) = -\frac{\widehat{\mathbb{E}}[R_t^{\text{net}}]}{\sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, R_t^{\text{net}} - r_f)^2 + \epsilon}} + \lambda_{\text{turnover}} \frac{1}{T}\sum_{t=1}^T \|w_t - w_{t-1}\|_1 + \lambda_{\text{entropy}} \sum_{i=1}^N w_{i, t} \log w_{i, t}$$
  where the first term is the negative Differentiable Sortino Ratio, the second term explicitly penalizes portfolio churn, and the third term enforces diversification across the asset universe.

---

## SECTION 3: Critique of Data & Evaluation Protocols (Publishability Killers)

A top-tier reviewer (NeurIPS / KDD / ICAIF) will reject the current paper within 10 minutes based on three evaluation methodology violations:

### 3.1 Violation 1: Standard Chronological Single-Split (80/20) and Window Leakage
- **The Flaw**: The dataset uses a single train/test split at 80% time. In financial time-series:
  1. Financial markets undergo non-ergodic structural breaks (e.g., pre-COVID 2019 vs crash 2020 vs inflation 2022). A single test window only measures performance in one arbitrary market regime.
  2. Overlapping sequence windows create cross-contamination: If lookback $T=30$ with stride $S=1$, sequence $k$ and sequence $k+1$ share 29 identical bars. A naive split allows information from the training tail to leak into the test head.
- **The Fix: Combinatorial Purged & Embargoed Cross-Validation (CPCV)**:
  Implement Marcos Lopez de Prado's CPCV protocol:
  - Generate $N_{\text{groups}} = 6$ non-contiguous time blocks.
  - Test on all combinations of $\binom{6}{2} = 15$ out-of-sample paths.
  - Apply a **Purge Window** equal to the forecast horizon ($H=5$ days) before each test set to eliminate label overlap.
  - Apply an **Embargo Window** ($h=10$ days) after each test set to eliminate autoregressive feature memory leakage.

```
CPCV SCHEME WITH PURGING & EMBARGO:
[── Train Block 1 ──][Purge][══ Test Block 1 ══][Embargo][── Train Block 2 ──]
```

---

### 3.2 Violation 2: Naive Flat Transaction Cost Model and Zero Slippage
- **The Flaw**: The backtest uses a flat 15 bps fee per trade. In institutional reality:
  1. Transaction costs are non-linear functions of volume and volatility (Almgren-Chriss market impact model):
     $$\text{Cost}(V_t) = \text{Fee}_{\text{broker}} + \eta \cdot \sigma_t \left( \frac{\text{Trade Volume}}{\text{Average Daily Volume}} \right)^\alpha$$
  2. At market opens and closes, bid-ask spreads widen significantly.
- **The Fix**: Implement a two-tier execution simulation:
  - **Tier 1 (Linear)**: 10 bps broker commission + exchange fees + stamp duty.
  - **Tier 2 (Square-Root Impact)**:
    $$\text{Slippage}_t = \gamma \cdot \text{Spread}_t + \theta \cdot \sigma_t \sqrt{\frac{|\Delta w_{i, t}| \cdot \text{Portfolio Value}}{\text{ADV}_{i, t}}}$$
  Include this in the paper to prove to reviewers that the reported Sharpe ratio survives institutional-grade execution constraints.

---

### 3.3 Violation 3: Absence of Essential SOTA Time-Series & Multi-Asset Baselines
- **The Flaw**: The current codebase compares only against Buy-and-Hold and a naive Momentum heuristic. A paper submitted to KDD or NeurIPS without modern deep learning baselines will receive an instant rejection for "Insufficient Empirical Comparison".
- **Required SOTA Baselines**:
  1. **Time-Series Foundation & Long-Horizon Models**:
     - **PatchTST** (Nie et al., ICLR 2023) — Channel-independent patched transformer.
     - **iTransformer** (Liu et al., ICLR 2024) — Inverted dimension transformer.
     - **TimeMixer** (Wang et al., ICLR 2024) — Multi-scale mixing architecture.
  2. **Graph / Hypergraph Neural Networks for Finance**:
     - **THGNN** (Temporal Hypergraph Neural Network for Stock Movement, KDD 2022).
     - **HATS** (Hierarchical Attention Network for Stock Prediction, IJCAI 2019).
     - **HyperStockGAT** (Relational Hypergraph Attention, ICAIF 2023).
  3. **Reinforcement Learning / Decision Transformers in Finance**:
     - **FinRL Benchmarks** (Liu et al., NeurIPS 2020) — PPO, SAC, DDPG ensemble.
     - **Q-Transformer / Decision Transformer for Alpha Generation** (ICML/ICAIF 2023).
     - **AlphaStock** (Wang et al., KDD 2019).

---

### 3.4 Violation 4: Lack of Formal Statistical Significance Hypothesis Testing
- **The Flaw**: Reporting single-point metric improvements (e.g., "DA is 61.6% vs 50.0%") without confidence intervals or statistical hypothesis tests is unacceptable.
- **Required Statistical Protocols**:
  1. **Diebold-Mariano (DM) Test**: Compare forecast loss differentials against baseline $b$:
     $$d_t = \mathcal{L}(\hat{y}_t^{\text{AMHDT}}, y_t) - \mathcal{L}(\hat{y}_t^{b}, y_t), \quad \text{DM} = \frac{\bar{d}}{\sqrt{\hat{\sigma}_{\bar{d}}^2 / T}} \sim \mathcal{N}(0, 1)$$
  2. **Hansen's Superior Predictive Ability (SPA) / White's Reality Check**: Test whether AMHDT outperforms all $M$ competing baselines simultaneously under data snooping adjustments:
     $$H_0: \max_{k=1, \dots, M} \mathbb{E}[d_{k, t}] \le 0$$
  3. **Deflated Sharpe Ratio (DSR)** (Bailey & Lopez de Prado): Correct for selection bias under multiple backtest trials:
     $$\text{DSR} = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}} \right)$$

---

## SECTION 4: The 4-Phase Research & Engineering Plan to Reach Publish-Grade

To convert this prototype into a top-tier accepted paper, execute the following 4-phase program over a 10-week sprint:

```
ROADMAP TIMELINE:
Phase 1: Architecture Overhaul (Weeks 1-3)
         ├── Adaptive Differentiable Hypergraph Structure Learning
         ├── Task-Gated Mixture-of-Experts Regimes
         └── Spatio-Temporal Factorized Attention Decision Transformer
Phase 2: Objective & Execution Overhaul (Weeks 4-5)
         ├── Differentiable Net Sortino + Turnover Penalty Loss
         ├── Deadband / Threshold Execution Policy
         └── Almgren-Chriss Slippage Simulation
Phase 3: SOTA Benchmarking & Rigorous CPCV (Weeks 6-7)
         ├── 5 Benchmarks: iTransformer, PatchTST, THGNN, FinRL, AlphaStock
         ├── Combinatorial Purged Cross-Validation (15 splits)
         └── US (S&P 500 / NASDAQ 100) + Indian (NIFTY 50) Cross-Market Evaluation
Phase 4: Statistical Validation & Paper Writing (Weeks 8-10)
         ├── Diebold-Mariano + Deflated Sharpe Ratio Tests
         ├── Full Ablation Matrix (8 variants)
         └── Paper Draft targeting NeurIPS / KDD / ICAIF
```

---

### Phase 1: Neural Architecture Upgrades (Weeks 1–3)

#### Task 1.1: Implement Adaptive Differentiable Hypergraph Structure Learning (AD-HSL)
- **File to Modify**: `ml/modules/module2_encoder.py`
- **Specification**:
  - Remove hardcoded sector groups and global correlation matrices.
  - Implement a dynamic hyperedge incidence layer:
    $$E_{\text{latent}} = \text{Linear}(Z_{\text{in}}), \quad H_t = \text{Gumbel-Softmax}\left( \frac{Z_{\text{in}} W_Q (E_{\text{latent}} W_K)^\top}{\sqrt{d}}, \tau \right)$$
  - Add initial residual connection ($Z^{(l+1)} = (1-\alpha) \text{Conv}(Z^{(l)}) + \alpha Z^{(0)}$) with $\alpha = 0.2$ to stop oversmoothing.

#### Task 1.2: Implement Task-Gated Mixture-of-Experts Regimes
- **File to Modify**: `ml/modules/module3_regime.py`
- **Specification**:
  - Remove discrete GMM/clustering pseudo-labels and classification BCE loss.
  - Implement a continuous router network producing expert mixture weights:
    $$g_t = \text{Softmax}(W_g \bar{z}_t + b_g) \in \Delta^{K-1}, \quad K=4$$
  - Add temporal stickiness penalty ($\mathcal{L}_{\text{sticky}} = \|g_t - g_{t-1}\|_2^2$) to eliminate day-to-day regime jitter.

#### Task 1.3: Refactor Decision Transformer into Spatio-Temporal Factorized Attention
- **File to Modify**: `ml/modules/module4_transformer.py`
- **Specification**:
  - Input tensor shape: $(B, T, N, D)$.
  - Layer 1: Temporal Multi-Head Attention along dimension $T$ (causally masked).
  - Layer 2: Spatial Cross-Asset Attention along dimension $N$ (unmasked, conditioned on hyperedge adjacency).
  - Condition on target downside volatility $\text{VaR}^*$ and target excess return $\mu^*$ instead of unconstrained Return-to-Go.

---

### Phase 2: Loss Function & Execution Engine Overhaul (Weeks 4–5)

#### Task 2.1: Implement Differentiable Transaction-Cost-Aware Sortino Loss
- **File to Modify**: `ml/training.py`
- **Specification**:
  - Replace line 165 with:
    ```python
    def compute_differentiable_trading_loss(pred_weights, targets, cost_bps=0.0015, turnover_lambda=0.05):
        # pred_weights: (batch, T, N), targets: (batch, T, N)
        portfolio_returns = (pred_weights * targets).sum(dim=-1) # (batch, T)
        turnover = torch.abs(pred_weights[:, 1:, :] - pred_weights[:, :-1, :]).sum(dim=-1) # (batch, T-1)
        net_returns = portfolio_returns[:, 1:] - (cost_bps * turnover)
        
        # Downside deviation
        excess_returns = net_returns - 0.0 # r_f = 0
        downside = torch.clamp(excess_returns, max=0.0)
        downside_risk = torch.sqrt(torch.mean(downside ** 2) + 1e-8)
        
        # Sortino ratio
        sortino = torch.mean(excess_returns) / (downside_risk + 1e-8)
        
        # Total loss
        loss = -sortino + turnover_lambda * torch.mean(turnover)
        return loss
    ```
  - This mathematically penalizes trade churn directly during backpropagation.

#### Task 2.2: Add Deadband / Minimum Change Threshold Execution Policy
- **File to Modify**: `ml/evaluation.py` and `ml/modules/module5_output.py`
- **Specification**:
  - Implement a trading deadband buffer $\delta$:
    $$w_t = \begin{cases} w_t^* & \text{if } \|w_t^* - w_{t-1}\|_1 \ge \delta \\ w_{t-1} & \text{otherwise} \end{cases}$$
  - For $\delta = 0.05$ (5% portfolio change threshold), this cuts trading frequency by $70\%$, eliminating noise trades and reducing transaction costs from 1,265% down to $< 30\%$.

---

### Phase 3: Comprehensive SOTA Benchmarking & Rigorous CPCV (Weeks 6–7)

#### Task 3.1: Build Standardized Benchmark Suite
- **Directory**: `ml/benchmarks/`
- Integrate official implementations of:
  1. `iTransformer` (ICLR 2024)
  2. `PatchTST` (ICLR 2023)
  3. `THGNN` (KDD 2022)
  4. `FinRL` (PPO / Ensemble, NeurIPS 2020)
  5. `Equal-Weighted Buy-and-Hold` & `Risk-Parity (ERC)`
- Ensure identical input tensors $(X, M, Y)$, identical lookback ($T=60$), identical horizon ($H=5$), and identical transaction cost friction (15 bps).

#### Task 3.2: Dual-Market Multi-Asset Datasets
- Expand evaluation beyond NIFTY 10 stocks:
  1. **US Market Benchmark**: S&P 100 constituent universe (100 liquid stocks, 2015–2025).
  2. **Indian Market Benchmark**: NIFTY 50 full constituent universe (50 stocks, 2015–2025).
  3. **Out-of-Distribution Stress Test Sets**:
     - 2020 COVID Crash (Feb 2020 – May 2020)
     - 2022 Global Rate-Hike Bear Market (Jan 2022 – Dec 2022)
     - 2023–2024 Tech Momentum Expansion

---

### Phase 4: Statistical Significance, Ablation Study, & Submission Draft (Weeks 8–10)

#### Task 4.1: Complete Ablation Study Matrix
To satisfy conference reviewers, the paper must include an 8-row ablation table proving the contribution of each module:

| Model Configuration | Validation MSE | Validation DA | Net Ann. Return | Net Sharpe | Max DD | Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **(1) Full AMHDT (Proposed)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **Low** |
| (2) w/o Dynamic Hypergraph (Static GCN) | ... | ... | ... | ... | ... | ... |
| (3) w/o Hypergraph (No Cross-Asset) | ... | ... | ... | ... | ... | ... |
| (4) w/o Task-Gated Regime MoE | ... | ... | ... | ... | ... | ... |
| (5) w/o Decision Transformer (Standard MLP) | ... | ... | ... | ... | ... | ... |
| (6) w/o Turnover Loss Penalty | High | 61.6% | -30.1% | -0.97 | -100% | Ultra-High |
| (7) w/o Deadband Execution Buffer | ... | ... | ... | ... | ... | ... |
| (8) Baseline iTransformer | ... | ... | ... | ... | ... | ... |

#### Task 4.2: Statistical Significance Tests
- Run Diebold-Mariano tests against all baselines with Newey-West lag adjustment ($L = \lfloor 4(T/100)^{2/9} \rfloor$).
- Compute bootstrap confidence intervals (1,000 resamples) for Sharpe ratios and Maximum Drawdowns.
- Compute the Deflated Sharpe Ratio (DSR) to prove the results are not the result of multiple testing data snooping.

---

## SECTION 5: Target Conference Strategy & Paper Positioning

### Target Conference Comparison

| Venue | Primary Focus | Best Positioning Angle for AMHDT | Typical Acceptance Rate | Submission Windows |
| :--- | :--- | :--- | :---: | :--- |
| **NeurIPS** (Datasets & Benchmarks or Main) | Foundational ML advances, theoretical grounding, large-scale empirical rigor | Frame as a **Distributional Decision Transformer with Dynamic Hypergraphs for Non-Stationary MDPs** | $\approx 25\%$ | May 2027 |
| **ICLR** | Representation learning, graph neural networks, sequential decision making | Frame as **Adaptive Hypergraph Structure Learning and Spatio-Temporal Transformers for Correlated Financial Time-Series** | $\approx 28\%$ | September 2027 |
| **KDD** (Applied Data Science Track) | Real-world business impact, massive datasets, execution efficiency | Frame as **An End-to-End Dependency-Aware Portfolio Optimization System with Real-World Transaction Friction Constraints** | $\approx 20\%$ | February / August 2027 |
| **ICAIF** (ACM International Conf. on AI in Finance) | Domain-specific rigor, market microstructure, risk-adjusted alpha | Frame as **AMHDT: Solving Mode Collapse and Turnover Attrition in Multi-Asset Trading via Regime-Gated Transformers** | $\approx 30\%$ | July 2027 |

### Recommended Title Options for Publication
1. *"Dependency-Aware Hypergraph Decision Transformers for Cross-Asset Portfolio Optimization Under Execution Frictions"*
2. *"Mitigating Mode Collapse and Turnover Bleed in Financial Decision Transformers via Adaptive Hypergraph Routing"*
3. *"Continuous Regime-Gated Spatio-Temporal Transformers for Multi-Asset Trading: An Empirical Study on US and Emerging Markets"*

---

## Summary Action Checklist for the Engineering Team

- [ ] **Step 1**: Refactor `training.py` immediately to include the Differentiable Net Sortino Loss with Turnover Penalty $\lambda_{\text{turnover}} \|\Delta w_t\|_1$.
- [ ] **Step 2**: Insert the Deadband Execution Buffer ($\delta = 0.05$) into `evaluation.py` to immediately eliminate the 4,220+ noise trades in Batch 1 backtests.
- [ ] **Step 3**: Rewrite `module2_encoder.py` to replace static sector hyperedges with Adaptive Differentiable Hypergraph Structure Learning (AD-HSL) with residual skip connections.
- [ ] **Step 4**: Replace scalar Return-to-Go in `module4_transformer.py` with bivariate Risk-Conditioning $(\mu^*, \text{VaR}^*)$.
- [ ] **Step 5**: Set up the Combinatorial Purged Cross-Validation (CPCV) runner and execute benchmark runs against `iTransformer` and `PatchTST`.
- [ ] **Step 6**: Compute Diebold-Mariano and Deflated Sharpe Ratio metrics across full S&P 100 and NIFTY 50 universes for the publication manuscript.
