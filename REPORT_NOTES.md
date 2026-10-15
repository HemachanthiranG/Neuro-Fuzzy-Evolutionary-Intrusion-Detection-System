# Technical Report Notes & Methodology Documentation

This document synthesizes the architectural decisions, theoretical methodology, mathematical formulations, experimental design, and reporting guidance for the **Evolutionary Neuro-Fuzzy Intrusion Detection System** on the NSL-KDD benchmark.

> **Crucial Rule on Metric Data:**
> In accordance with project reproducibility rules, exact performance metrics are never hardcoded in this documentation. All quantitative values must be read directly from the generated result files:
> - Multi-model ablation benchmark: [`results/ablation_summary.csv`](results/ablation_summary.csv)
> - Win / Loss / Tie statistical audit: [`results/win_loss_table.csv`](results/win_loss_table.csv)
> - Paired Wilcoxon hypothesis tests: [`results/statistical_tests.json`](results/statistical_tests.json)
> - Suspicious-band record triage & composition: [`results/fuzzy_decision_stats.json`](results/fuzzy_decision_stats.json)
> - Linguistic Mamdani rule bases: [`results/fuzzy_rules.json`](results/fuzzy_rules.json)
> - Granular tabular runs: [`results/metrics_all_runs.csv`](results/metrics_all_runs.csv)
> - Per-class metrics: [`results/per_class_metrics.csv`](results/per_class_metrics.csv)
> - Consensus feature subsets: [`results/selected_features.json`](results/selected_features.json)

---

## 1. Architectural & Theoretical Methodology

### 1.1 Multi-Objective Feature Selection Formulation
The system models feature selection as a combinatorial multi-objective optimization problem over the 41 original network attributes of NSL-KDD. The search seeks a sparse feature subset that maximizes intrusion classification utility, penalizes feature cardinality, and actively suppresses false alarms:

$$\text{Fitness}(\mathbf{c}) = \text{Macro-F1}_{\text{val}}(\mathbf{c}) - \alpha \cdot \left(\frac{\sum_{i=1}^{41} c_i}{41}\right) - \beta \cdot \text{FPR}_{\text{val}}(\mathbf{c})$$

where:
- $\mathbf{c} \in \{0, 1\}^{41}$ is the binary chromosome vector.
- $\text{Macro-F1}_{\text{val}}(\mathbf{c})$ is the validation macro-averaged F1 score evaluated using a fast proxy Artificial Neural Network (ANN).
- $\text{FPR}_{\text{val}}(\mathbf{c}) = \frac{\text{FP}_{\text{val}}}{\text{FP}_{\text{val}} + \text{TN}_{\text{val}}}$ is the binary false positive rate on the validation split.
- $\alpha = 0.02$ is the feature parsimony penalty coefficient.
- $\beta = 0.30$ is the false positive rate penalty coefficient configured in `config.yaml`.

---

### 1.2 Fuzzy-Adaptive Genetic Algorithm Controller

Standard genetic algorithms suffer from premature convergence or disruptive mutation when static crossover ($p_c$) and mutation ($p_m$) probabilities are maintained throughout the search. To overcome this, a pure NumPy **Mamdani Fuzzy Inference System** dynamically adapts genetic operators at every generation based on the state of the population:

#### 1.2.1 State Inputs
1. **Population Diversity ($D \in [0, 1]$):** Computed as the normalized mean pairwise Hamming distance across all chromosomes in the current population:
   $$D = \frac{2}{P(P-1) \cdot L} \sum_{i=1}^{P-1} \sum_{j=i+1}^P \sum_{k=1}^L |c_{i,k} - c_{j,k}|$$
   where $P$ is the population size and $L=41$ is the chromosome length.
2. **Fitness Stagnation ($S \in [0, 1]$):** Measures the number of elapsed generations since the best global fitness improved, normalized by early stopping patience:
   $$S = \min\left(1.0, \frac{\text{generations\_since\_improvement}}{\text{early\_stop\_patience}}\right)$$

#### 1.2.2 Linguistic Terms & Membership Functions
Both inputs are mapped to three triangular fuzzy sets (**Low**, **Medium**, **High**) bounded over $[0, 1]$:
$$\mu_{\text{tri}}(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$

#### 1.2.3 Output Operator Defuzzification
The controller produces two continuous parameters defuzzified via center-of-gravity (centroid) defuzzification:
- **Mutation Probability ($p_m \in [0.005, 0.10]$):** Elevated when diversity is low or stagnation is high; reduced during periods of healthy progress.
- **Crossover Probability ($p_c \in [0.60, 0.95]$):** Elevated when diversity is healthy to promote recombination; dampened during severe stagnation.

#### 1.2.4 Complete 9-Rule Mamdani Controller Base
```text
Rule 1: IF Diversity is Low    AND Stagnation is High   THEN Mutation is High,   Crossover is Low
Rule 2: IF Diversity is Low    AND Stagnation is Medium THEN Mutation is High,   Crossover is Medium
Rule 3: IF Diversity is Low    AND Stagnation is Low    THEN Mutation is Medium, Crossover is High
Rule 4: IF Diversity is Medium AND Stagnation is High   THEN Mutation is High,   Crossover is Medium
Rule 5: IF Diversity is Medium AND Stagnation is Medium THEN Mutation is Medium, Crossover is High
Rule 6: IF Diversity is Medium AND Stagnation is Low    THEN Mutation is Low,    Crossover is High
Rule 7: IF Diversity is High   AND Stagnation is High   THEN Mutation is Medium, Crossover is Medium
Rule 8: IF Diversity is High   AND Stagnation is Medium THEN Mutation is Low,    Crossover is High
Rule 9: IF Diversity is High   AND Stagnation is Low    THEN Mutation is Low,    Crossover is High
```

---

### 1.3 Gaussian Neuro-Fuzzy Input Layer

Network traffic features exhibit continuous distributions with severe skewness and multimodal characteristics. The neuro-fuzzy input layer transforms raw numeric attributes into interpretable linguistic membership degrees:

1. **Fuzzy C-Means (FCM) Clustering:** For each active continuous feature $j$, 1D FCM clustering is executed strictly on the training partition to discover $C=3$ cluster centers ($c_1 < c_2 < c_3$) representing **Low**, **Medium**, and **High**.
2. **Gaussian Membership Functions:** For each linguistic term $k \in \{1, 2, 3\}$, the membership degree of feature value $x$ is defined as:
   $$\mu_k(x) = \exp\left(-\frac{(x - c_k)^2}{2\sigma_k^2}\right)$$
   where width $\sigma_k$ is derived from inter-cluster distances $\sigma_k = \frac{|c_{k+1} - c_k|}{2\sqrt{2\ln 2}}$.
3. **Robust Fallbacks:** Features with near-zero variance, extreme skewness, or binary indicator structures bypass FCM and use empirical quantile partitioning ($[Q_{0.10}, Q_{0.50}, Q_{0.90}]$) to guarantee non-zero widths ($\sigma \ge 10^{-4}$) and eliminate division-by-zero or NaN artifacts.
4. **Input Representation:** The final neural input vector is formed by concatenating scaled continuous features, one-hot encoded categorical columns, and the computed fuzzy membership degrees:
   $$\mathbf{x}_{\text{final}} = [\mathbf{x}_{\text{scaled}}, \mathbf{x}_{\text{onehot}}, \boldsymbol{\mu}_{\text{fuzzy}}]$$
5. **Persistence:** All fitted cluster centers and bandwidth parameters are serialized directly inside `models/preprocessor.joblib`, ensuring identical transformation at inference time.

---

### 1.4 Post-ANN Mamdani Fuzzy Decision Layer

Standard neural networks force hard classification via argmax over the softmax vector $\hat{\mathbf{y}}$, obscuring borderline decisions where multiple classes have non-negligible probabilities. The post-ANN decision layer implements an interpretable Mamdani inference engine for threat triage:

#### 1.4.1 Decision Inputs
1. **Model Confidence ($C \in [0, 1]$):** The maximum predicted class probability:
   $$C = \max_{k \in \{1,\dots,5\}} \hat{y}_k$$
2. **Normal Probability ($P_N \in [0, 1]$):** The predicted probability of the benign class:
   $$P_N = \hat{y}_{\text{Normal}}$$
3. **Normalized Prediction Entropy ($H \in [0, 1]$):** Quantifies decision uncertainty across the 5 output classes:
   $$H = -\frac{1}{\ln 5} \sum_{k=1}^5 \hat{y}_k \ln(\hat{y}_k + \epsilon)$$

#### 1.4.2 Decision Rule Base & Alert Levels
The inference engine maps inputs to an **Attack Severity Score** in $[0, 1]$ and assigns a 3-level operational alert status:
- **Normal:** Low severity, high confidence in benign traffic.
- **Attack:** High severity, clear malicious signature.
- **Suspicious:** Intermediate severity, elevated entropy, or conflicting confidence. Triaged for secondary inspection.

#### 1.4.3 Complete 9-Rule Decision Base
```text
Rule 1: IF Confidence is High   AND P(Normal) is High                         THEN Severity is Normal
Rule 2: IF Confidence is Medium AND P(Normal) is High AND Entropy is Low     THEN Severity is Normal
Rule 3: IF Confidence is High   AND P(Normal) is Low                          THEN Severity is Attack
Rule 4: IF Confidence is Medium AND P(Normal) is Low  AND Entropy is Low     THEN Severity is Attack
Rule 5: IF Confidence is Low                          AND Entropy is High    THEN Severity is Suspicious
Rule 6: IF Confidence is Medium                       AND Entropy is High    THEN Severity is Suspicious
Rule 7: IF P(Normal) is Medium                                               THEN Severity is Suspicious
Rule 8: IF Entropy is High                                                   THEN Severity is Suspicious
Rule 9: IF Confidence is Medium AND P(Normal) is Low  AND Entropy is Medium THEN Severity is Attack
```

#### 1.4.4 Validation-Only Band Tuning & 5-Class Label Mapping
- The boundary band $[\theta_{\text{low}}, \theta_{\text{high}}]$ governing the **Suspicious** alert is tuned strictly on the validation partition via grid search to maximize $\text{Macro-F1}_{\text{val}} - \beta \cdot \text{FPR}_{\text{val}}$.
- Records with severity score $S \in [\theta_{\text{low}}, \theta_{\text{high}}]$ are flagged with the operational alert level **Suspicious**.
- **Labeling in 5-Class Evaluation:** In the 5-class evaluation, Suspicious records are assigned their neural network argmax prediction (`hard_preds`). On the test set (`KDDTest+`), exactly **0.0000% of records (0 / 22,544)** differ between `fuzzy_adjusted_class` and `hard_preds` for both Variant 2 (`Baseline ANN + Fuzzy-decision`) and Variant 7 (`GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision`). The decision layer's 3-level alert (`Normal`, `Suspicious`, `Attack`) and continuous severity score ($[0, 1]$) function as an operational triage/escalation mechanism for human-in-the-loop review rather than altering 5-class categorical assignments.

---

## 2. Leakage Prevention Protocol

Data leakage compromises empirical validity in intrusion detection research. This codebase implements five structural safeguards verified by automated pytest assertions:

1. **Partition Segregation:** Preprocessing transformers (`StandardScaler`, `OneHotEncoder`, FCM cluster centers) are fitted exclusively on the 80% training split of `KDDTrain+`. Validation and test partitions are transformed strictly out-of-sample.
2. **Zero Test Set Contamination:** The held-out benchmark (`KDDTest+`) is evaluated exactly once per model architecture after training terminates. The test set is never accessed during GA fitness evaluation, hyperparameter search, or decision layer tuning.
3. **Target & Metadata Exclusion:** Target labels and the benchmark difficulty index (column 43) are stripped before feature extraction.
4. **Deterministic Reproducibility:** Global random seeds lock state across Python `random`, `numpy.random`, and `tensorflow.random`.
5. **Algorithmic Class Balancing:** Class imbalance is addressed through cost-sensitive cross-entropy weights rather than synthetic oversampling (e.g., SMOTE) on evaluation data.

---

## 3. Ablation Study & Model Matrix

The 9 evaluated architectures systematically isolate each component:

| Model Variant | Feature Selector | Input Layer | Classifier | Decision Layer | Primary Isolation Objective |
|---|---|---|---|---|---|
| **Baseline ANN** | None (All 41) | Standard Scaled/Encoded | Keras MLP | Hard Argmax | Benchmark control for all features |
| **GA + ANN** | Standard GA | Standard Scaled/Encoded | Keras MLP | Hard Argmax | Isolates basic evolutionary feature selection |
| **GA(adaptive) + ANN** | Fuzzy-Adaptive GA | Standard Scaled/Encoded | Keras MLP | Hard Argmax | Isolates the Mamdani GA parameter controller |
| **GA + Fuzzy-input + ANN** | Standard GA | Scaled + Gaussian Memberships | Keras MLP | Hard Argmax | Isolates Gaussian neuro-fuzzy input expansion |
| **GA(adaptive) + Fuzzy-input + ANN** | Fuzzy-Adaptive GA | Scaled + Gaussian Memberships | Keras MLP | Hard Argmax | Tests synergy of adaptive search and fuzzy inputs |
| **GA(adaptive) + Fuzzy-input + ANN + Fuzzy-decision** | Fuzzy-Adaptive GA | Scaled + Gaussian Memberships | Keras MLP | Mamdani Triage | Full proposed neuro-fuzzy architecture |
| **MI + ANN** | Mutual Information | Standard Scaled/Encoded | Keras MLP | Hard Argmax | Information-theoretic filter control |
| **RF + ANN** | Random Forest Gini | Standard Scaled/Encoded | Keras MLP | Hard Argmax | Tree-based importance filter control |
| **Baseline ANN + Fuzzy-decision** | None (All 41) | Standard Scaled/Encoded | Keras MLP | Mamdani Triage | Isolates post-ANN decision triage on full features |

---

## 4. How to Interpret Evaluation Results

### 4.1 NSL-KDD Zero-Day Distribution Shift & Attack Taxonomy
When evaluating models on `KDDTest+`, performance metrics exhibit lower values than on `KDDTrain+`. This divergence is an intentional property of the NSL-KDD benchmark:
- **Attack Variety:** `KDDTrain+` contains 22 distinct attack types (+ normal), whereas `KDDTest+` contains 37 distinct attack types (+ normal).
- **R2L Attack Types in Training:** `KDDTrain+` contains exactly **8 R2L attack types**:
  `ftp_write`, `guess_passwd`, `imap`, `multihop`, `phf`, `spy`, `warezclient`, `warezmaster`.
- **17 Unseen Zero-Day Attack Variants:** Exactly 17 attack types appear in `KDDTest+` that are completely absent from `KDDTrain+`, spanning all 4 attack categories:
  * **DoS (5 types):** `apache2`, `mailbomb`, `processtable`, `udpstorm`, `worm`
  * **Probe (2 types):** `mscan`, `saint`
  * **R2L (7 types):** `httptunnel`, `named`, `sendmail`, `snmpgetattack`, `snmpguess`, `xlock`, `xsnoop`
  * **U2R (3 types):** `ps`, `sqlattack`, `xterm`
- **Class Proportions:** The proportion of **Remote-to-Local (R2L)** attacks surges from **0.79% in training** to **12.80% in testing**, while benign **Normal** traffic declines from **53.46% to 43.08%**. Test metrics evaluate out-of-distribution generalization to previously unseen exploits.

### 4.2 Security-Critical Trade-Offs: Detection Rate vs. False Positive Rate
In operational security environments, standard accuracy can be misleading due to benign majority dominance. Key metrics:
- **Attack Detection Rate (ADR / Recall):** Proportion of true malicious traffic successfully intercepted. Higher is better.
- **False Negative Rate (FNR = 1 - ADR):** Proportion of intrusions that bypass detection undetected. Lower is better.
- **False Positive Rate (FPR):** Proportion of legitimate benign network connections mistakenly flagged as threats. Lower is better.

### 4.3 Suspicious Band Operational Significance & R2L Decomposition
An inspection of `results/fuzzy_decision_stats.json` reveals the empirical purpose of the validation-tuned fuzzy decision layer:
- In the validation partition, the Suspicious band flags a small volume of ambiguous records dominated by Normal flows (over 90% Normal).
- In the held-out test set (`KDDTest+`), however, the Suspicious band flags over 53% R2L network attacks across both decision variants.
- **Seen vs. Unseen R2L Decomposition in Suspicious Band:**
  * **Proposed Model (Variant 7, `GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision`):**
    - Total Suspicious R2L test records: **719 records** (53.26% of all 1,350 Suspicious records).
    - **Unseen Zero-Day Attacks: 551 records (76.63%)** (`snmpguess`: 326, `snmpgetattack`: 154, `httptunnel`: 67, `named`: 3, `xlock`: 1).
    - **Seen Known Attacks: 168 records (23.37%)** (`warezmaster`: 161, `multihop`: 3, `guess_passwd`: 2, `imap`: 1, `ftp_write`: 1).
  * **Ablation Control (Variant 2, `Baseline ANN + Fuzzy-decision`):**
    - Total Suspicious R2L test records: **798 records** (53.49% of all 1,492 Suspicious records).
    - Seen Known Attacks: 771 records (96.62%).
    - Unseen Zero-Day Attacks: 27 records (3.38%).
- **Key Insight:** In the proposed model, over three-quarters (76.63%) of the Suspicious R2L test records are completely unseen zero-day attacks. The fuzzy-adaptive GA feature selection combined with neuro-fuzzy inputs pushes novel R2L attacks into high entropy and lower confidence, enabling the Mamdani decision layer to isolate them into the Suspicious band for secondary review.

---

## 5. Statistical Testing & Win/Loss Audit Protocol

To avoid drawing conclusions from stochastic variance, multi-seed runs execute formal hypothesis testing and audited programmatic comparisons:
- **Paired Hypothesis Tests:** Paired Wilcoxon signed-rank test comparing the Full Proposed Model against Baseline ANN and GA+ANN on Macro-F1, FPR, Accuracy, and Attack Detection Rate.
- **Fallback:** Paired Student's t-test with Cohen's d effect size calculation if paired differences are zero-variance or sample sizes prohibit rank-sum computation.
- **Power Disclaimer:** When evaluating with small seed counts ($n \le 5$), statistical power is constrained; failure to reject the null hypothesis ($p \ge 0.05$) should be interpreted cautiously and verified via multi-seed replication. All test statistics, p-values, and effect sizes are saved in [`results/statistical_tests.json`](results/statistical_tests.json).
- **Win/Loss Decision Logic:**
  1. **Insufficient Data Guard:** Requires at least $n \ge 3$ seeds before declaring `TIE`. For runs with fewer seeds ($n < 3$), metrics within the tolerance threshold are explicitly labeled `'INSUFFICIENT DATA (n<3)'`.
  2. **Standard Deviation Guard:** Sample standard deviations are never applied as a tolerance margin when $n = 1$.
  3. **Noise Guard on Latency:** Latency differences smaller than 10% are excluded from wins/losses and labeled `'WITHIN NOISE'`.

---

## 6. Known Limitations

1. **Benchmark Substratum Age:** While NSL-KDD eliminates duplicate record bias present in KDD Cup 99, its underlying packet captures reflect legacy 1998 network protocols. Modern encrypted traffic (TLS 1.3, QUIC, HTTP/3) requires validation on newer datasets (e.g., CIC-IDS-2017, UNSW-NB15).
2. **Extreme Minority Scarcity:** User-to-Root (U2R) attacks comprise only 42 records in `KDDTrain+` (0.04%), limiting neural boundary formation for privilege escalation detection.
3. **Input Dimensionality Expansion:** Gaussian fuzzy membership terms expand the feature dimensionality fed into the first hidden layer, partially offsetting the storage gains of feature pruning.
4. **FPR Trade-Off:** The aggressive feature pruning and fuzzy input expansion lead to an increased false positive rate on benign traffic, requiring threshold tuning based on organizational risk tolerance.
5. **5-Class Metric Invariance:** Because Suspicious records are mapped to their argmax neural class in standard multi-class evaluation, the decision layer functions as an operational triage/escalation flag rather than a direct multi-class accuracy enhancer.

---

## 7. Recommendations for Academic Report Sections

- **Section 1: Introduction:** Motivate automated feature selection and neuro-fuzzy representation in high-throughput network intrusion detection.
- **Section 2: Related Work:** Contrast evolutionary wrapper methods with filter selectors and explain the role of fuzzy logic in managing classifier uncertainty.
- **Section 3: Methodology:** Detail the two-tier architecture: binary chromosome representation, Mamdani adaptive controller, FCM membership fitting, neural MLP, and post-ANN decision triage.
- **Section 4: Experimental Setup:** Detail dataset preparation, stratified splitting, attack taxonomies, hardware environment, and zero-leakage safeguards.
- **Section 5: Results & Discussion:**
  - Tabulate primary multi-metric results from [`results/ablation_summary.csv`](results/ablation_summary.csv).
  - Review the win/loss distribution in [`results/win_loss_table.csv`](results/win_loss_table.csv).
  - Analyze the statistical hypothesis tests in [`results/statistical_tests.json`](results/statistical_tests.json).
  - Contrast convergence curves (`results/plots/fuzzy_2_ga_convergence_comparison.png`) and Pareto efficiency (`results/plots/fuzzy_4_pareto_frontier.png`).
  - Examine Suspicious band composition and R2L triage from [`results/fuzzy_decision_stats.json`](results/fuzzy_decision_stats.json).
- **Section 6: Threats to Validity:** Discuss the zero-day attack distribution shift between `KDDTrain+` and `KDDTest+`, and detail the verification of test leakage prevention.
- **Section 7: Conclusion:** Summarize trade-offs between feature parsimony, attack detection rate, and false positive rates.
