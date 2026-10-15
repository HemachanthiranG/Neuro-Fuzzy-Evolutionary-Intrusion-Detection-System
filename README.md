# Evolutionary Neuro-Fuzzy Intrusion Detection System (NSL-KDD)

A modular machine learning framework implementing an **Evolutionary Neuro-Fuzzy Intrusion Detection System (IDS)** on the benchmark **NSL-KDD dataset**.

The system integrates:
1. **Fuzzy-Adaptive Genetic Algorithm (GA)** for intelligent search across the 41 original network traffic features.
2. **Gaussian Neuro-Fuzzy Input Preprocessing** fitting fuzzy c-means (FCM) linguistic membership terms on training data.
3. **Deep Artificial Neural Network (Keras MLP)** cost-sensitively trained to classify network flows into 5 canonical categories (**Normal**, **DoS**, **Probe**, **R2L**, **U2R**).
4. **Post-ANN Fuzzy Decision Layer** implementing an interpretable Mamdani inference system for threat triage, defuzzified severity scoring, and operational alert escalation (**Normal**, **Suspicious**, **Attack**).
5. **Interactive Streamlit Web Dashboard** featuring multi-seed evaluation, publication plot galleries, real-time threat inference, and live fuzzy membership degree inspection.

---

## Evaluated Architectural Variants (Ablation Matrix)

To systematically isolate the individual contributions of the evolutionary search, fuzzy parameter controllers, fuzzy input representations, and fuzzy decision triage, the framework evaluates 9 architectural variants under identical training and evaluation budgets:

1. **Baseline ANN (All Features):** Full 41 original network traffic features (122 one-hot/scaled columns).
2. **GA + ANN:** Standard genetic algorithm feature selection with static crossover and mutation rates.
3. **GA(adaptive) + ANN:** Evolutionary search steered by a Mamdani fuzzy controller dynamically adjusting crossover and mutation rates from population diversity and fitness stagnation.
4. **GA + Fuzzy-input + ANN:** Standard GA selecting original features augmented with Gaussian membership functions fitted on numeric attributes.
5. **GA(adaptive) + Fuzzy-input + ANN:** Fuzzy-adaptive GA combined with Gaussian fuzzy input representations.
6. **GA(adaptive) + Fuzzy-input + ANN + Fuzzy-decision (Full Proposed Model):** The complete end-to-end neuro-fuzzy pipeline including evolutionary selection, fuzzy input representation, deep neural classifier, and post-ANN Mamdani decision triage.
7. **MI + ANN:** Mutual Information filter baseline selecting top-$k$ features matched to the GA cardinality budget.
8. **RF + ANN:** Random Forest Gini importance filter baseline selecting top-$k$ features matched to the GA cardinality budget.
9. **Baseline ANN + Fuzzy-decision:** Full 41-feature baseline ANN paired with the post-ANN fuzzy decision layer to isolate the decision mechanism independently from feature selection.

---

## Zero-Leakage Architecture & Experimental Integrity

In strict compliance with empirical evaluation protocols:
- **Zero Test Set Contamination:** All membership function parameters (FCM cluster centers and widths), GA fitness evaluations, neural network weights, early stopping, and fuzzy decision layer thresholds/bands are fitted or tuned **exclusively on the training and validation partitions of `KDDTrain+`**. The held-out benchmark (`KDDTest+`) is evaluated exactly once per trained model.
- **Automated Anti-Leakage Verification:** Pytest test cases explicitly verify function signatures and call graphs, ensuring test arrays are never passed into genetic search or decision layer tuning routines.
- **Strict Preprocessing Hygiene:** Categorical encoders (`OneHotEncoder`) and continuous standardizers (`StandardScaler`) fit strictly on the training partition.
- **No Target Leakage:** Neither the attack category label nor the benchmark difficulty score is ever passed as an input feature.
- **Original-to-Encoded Deterministic Mapping:** Binary chromosomes operate directly on the 41 original domain features. Activating an original feature automatically activates all associated one-hot encoded sub-features.
- **Cost-Sensitive Class Rebalancing:** Training loss utilizes inverse class weights calculated solely from training class frequencies to mitigate severe minority class representation (e.g., U2R at <0.1% of training instances).

---

## Directory Layout

```text
├── app/
│   ├── streamlit_app.py         # Streamlit dashboard (Ablation Analysis & Threat Predictor)
│   └── components/
│       ├── ui.py                # Visual styling, dark theme tokens, KPI cards, badge helpers
│       └── inference.py         # Dynamic seed resolution, test sampling, manual & batch prediction
├── data/
│   ├── raw/                     # KDDTrain+.txt and KDDTest+.txt benchmark files
│   └── processed/               # Cached preprocessed arrays and partition indices
├── models/                      # Saved .keras models, preprocessor.joblib, and feature masks per seed
├── results/
│   ├── ablation_summary.csv     # Multi-metric mean ± std across all 9 model variants
│   ├── win_loss_table.csv       # Programmatic statistical win/loss/tie audit vs Baseline ANN
│   ├── statistical_tests.json   # Paired Wilcoxon signed-rank and Cohen's d effect sizes
│   ├── fuzzy_decision_stats.json# Validation and test record distributions in the Suspicious band
│   ├── fuzzy_rules.json         # Complete Mamdani rule bases for GA controller and decision layer
│   ├── metrics_all_runs.csv     # Granular per-seed evaluation records across all models
│   ├── confusion_matrices/      # Normalized confusion matrix heatmaps and raw .npy files
│   └── plots/                   # 300 DPI publication diagnostic figures
├── src/
│   ├── __init__.py
│   ├── config.py                # Strongly-typed configuration schema
│   ├── fuzzy.py                 # Pure NumPy FCM, Gaussian input layer, GA controller, decision layer
│   ├── fuzzy_plots.py           # 300 DPI publication plots for fuzzy components
│   ├── plots.py                 # Core diagnostic plotting suite and automated regenerator
│   ├── ga.py                    # Pure NumPy GA with tournament, uniform crossover, bit-flip, elitism
│   ├── ann.py                   # Keras MLP model builder, callbacks, latency profiler
│   ├── baselines.py             # Filter selectors (Mutual Information, Random Forest)
│   ├── data_loader.py           # Data ingest and 40-attack-to-5-class taxonomy mapper
│   ├── preprocessing.py         # Leakage-free scaling, one-hot encoding, chromosome indexing
│   ├── evaluate.py              # Multi-metric scoring and confusion matrix evaluation
│   ├── run_experiments.py       # Orchestration runner with automatic plot rebuilding
│   └── utils.py                 # Deterministic seed locking, logging, timer, JSON/joblib I/O
├── tests/
│   ├── test_pipeline.py         # Core pipeline, schema, and leakage regression tests
│   └── test_fuzzy.py            # Neuro-fuzzy unit tests, bounds checks, and zero-leakage assertions
├── config.yaml                  # System hyperparameters, fuzzy rules, and execution modes
├── requirements.txt             # Pinned dependencies for Python 3.10/3.11 CPU
├── PROGRESS.md                  # Phase execution tracking log
├── REPORT_NOTES.md              # Technical methodology, rule formulations, and empirical guidance
└── README.md
```

---

## Installation & Setup

Verified on **Python 3.10.11 CPU** and **Python 3.11**.

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Pinned Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Acquire NSL-KDD Benchmark
Place `KDDTrain+.txt` and `KDDTest+.txt` into `data/raw/`, or run the acquisition script with Kaggle credentials:
```powershell
python -m src.download_data
```

---

## Execution Commands

### Fast Verification Run (Single Seed)
Executes a quick end-to-end verification across all 9 model variants on 1 seed with reduced GA generations and training epochs, evaluating on held-out test data, updating result files, and rebuilding all figures:
```powershell
python -m src.run_experiments --mode fast --seeds 1
```

### Multi-Seed Production Run (5 Seeds)
Executes the full experimental protocol across 5 random seeds (`[42, 123, 999, 2024, 777]`), executing 25 generations of fuzzy-adaptive GA search with 30 individuals, full training of all 9 architectures, multi-seed aggregation, Wilcoxon signed-rank significance tests, win/loss audit, and automated 300 DPI plot regeneration:
```powershell
python -m src.run_experiments --mode full --seeds 5
```

### Launch Interactive Web Dashboard
Launches the Streamlit application on local port 8501:
```powershell
streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in any web browser.

### Run Pytest Test Suite
Executes the complete 18-test automated verification suite:
```powershell
python -m pytest -v
```

---

## Streamlit Application Overview

The dashboard provides two primary interfaces driven entirely by dynamic result files:

### Tab 1: Results & Comparison
- **Dynamic Seed Selector:** Allows toggling between the aggregated multi-seed mean view and individual seed performance metrics discovered automatically from `results/` and `models/`.
- **Honest Performance Evaluation:** Dynamically computes and displays exact metric gains, trade-offs, and triage statistics directly from `results/ablation_summary.csv` and `results/fuzzy_decision_stats.json`.
- **Ablation Matrix:** Full tabular comparison of all 9 models reporting accuracy, macro-F1, attack detection rate, false positive rate (test and validation), false negative rate, original features used, encoded inputs, and latency.
- **Statistical Win/Loss Table:** Programmatically audit wins, ties (within one standard deviation), and losses against Baseline ANN.
- **Paired Hypothesis Testing:** Wilcoxon signed-rank and paired t-test results reporting test statistics, p-values, Cohen's d effect sizes, and sample size power disclaimers from `results/statistical_tests.json`.
- **Publication Plot Galleries:** Interactive displays of both core evolutionary plots and high-resolution fuzzy diagnostic figures.
- **Interpretable Rule Inspector:** Formatted viewing of the active Mamdani linguistic rules for the GA controller and post-ANN decision layer.

### Tab 2: Live Threat Predictor
- **Multi-Model Inference:** Select between the Full Proposed Model (`GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision`), `GA+ANN`, or `Baseline ANN`.
- **Operational Threat Categorization:** Outputs the standard 5-class prediction, post-ANN fuzzy adjusted label, 3-level alert status (**Normal**, **Suspicious**, **Attack**), and defuzzified attack severity score ($[0, 1]$).
- **Linguistic Membership Visualizer:** Interactive Plotly bar chart rendering real-time Low, Medium, and High Gaussian membership degrees for key network features.
- **Three Input Modes:**
  1. *Test Set Benchmark Sampler:* Select real network flows from `KDDTest+` filtered by attack category.
  2. *Interactive Manual Input:* Custom flow parameter entry across Basic, Content, Time-Based Traffic, and Host-Based Traffic groups.
  3. *Batch CSV Predictor:* Upload arbitrary connection CSVs for batch classification and downloadable audited reports.

---

## Result Files & Data Provenance

All quantitative performance figures are persisted in `results/` and must be inspected directly from the artifacts:

- **Comprehensive Model Benchmark:** [`results/ablation_summary.csv`](results/ablation_summary.csv)
- **Win / Loss / Tie Audit:** [`results/win_loss_table.csv`](results/win_loss_table.csv)
- **Wilcoxon Hypothesis Tests:** [`results/statistical_tests.json`](results/statistical_tests.json)
- **Tuned Suspicious Band & Class Triage:** [`results/fuzzy_decision_stats.json`](results/fuzzy_decision_stats.json)
- **Mamdani Fuzzy Rule Base:** [`results/fuzzy_rules.json`](results/fuzzy_rules.json)
- **Granular Run Metrics:** [`results/metrics_all_runs.csv`](results/metrics_all_runs.csv)
- **Consensus Feature Subsets:** [`results/selected_features.json`](results/selected_features.json)

---

## Honest Scientific Assessment

An objective examination of experimental outputs indicates a nuanced trade-off:

1. **Where the Proposed Model Excels (Empirical Wins):**
   - **Attack Detection Rate (ADR):** Substantially increases detection rate on unseen test attacks compared to the full-feature Baseline ANN.
   - **False Negative Rate (FNR):** Meaningfully reduces the rate of missed intrusions, ensuring fewer malicious vectors bypass detection.
   - **Feature Dimensionality Reduction:** Prunes more than 60% of original network features, operating on a concise, domain-relevant subset.
   - **Zero-Day Attack Triage:** The validation-tuned fuzzy decision layer triages high-entropy connection records into the Suspicious band, where over half of flagged test records consist of R2L network exploits (with over 75% of them belonging to novel zero-day attack variants absent from training).

2. **Where the Proposed Model Faces Trade-Offs (Empirical Costs):**
   - **False Positive Rate (FPR):** Incurs an elevated false positive rate on both validation and test partitions relative to the full-feature baseline.
   - **Test Macro-F1 & Accuracy:** Test macro-F1 and overall accuracy reflect a slight decline due to increased false alarms on benign test traffic.
   - **Input Dimensionality:** Gaussian membership term expansion introduces additional input dimensions into the first neural layer.
   - **5-Class Metric Invariance:** In 5-class evaluation, exactly 0.00% of test records have fuzzy_adjusted_class != argmax (0 / 22,544 records). Suspicious records are labeled with their neural argmax prediction; the decision layer serves primarily as an operational human-in-the-loop escalation mechanism rather than altering standard multi-class categorical assignments.
