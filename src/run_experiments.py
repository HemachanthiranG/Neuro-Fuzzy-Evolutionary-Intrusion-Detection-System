"""End-to-end experiment orchestration and CLI runner for NSL-KDD Feature Selection.

Supports complete 9-model ablation study:
1. Baseline ANN (all 41 features)
2. Baseline ANN + Fuzzy-decision (isolates fuzzy decision layer on all features)
3. GA + ANN (Standard GA feature selection)
4. GA(adaptive) + ANN (Fuzzy-adaptive GA feature selection)
5. GA + Fuzzy-input + ANN (Standard GA + Gaussian membership neuro-fuzzy input layer)
6. GA(adaptive) + Fuzzy-input + ANN (Adaptive GA + Neuro-fuzzy input layer)
7. GA(adaptive) + Fuzzy-input + ANN + Fuzzy-decision (Full Proposed Model)
8. MI + ANN (Mutual Information top-k matched)
9. RF + ANN (Random Forest top-k matched)

Generates:
- results/ablation_summary.csv (mean ± std per metric across seeds, including val_fpr and test_fpr)
- results/statistical_tests.json (paired Wilcoxon signed-rank tests vs Baseline and GA+ANN)
- results/win_loss_table.csv and win_loss_table.json
- results/fuzzy_decision_stats.json (per-seed tuned suspicious bands, validation f1/fpr before/after, composition)
- results/metrics_all_runs.csv, results/metrics_summary.json, results/per_class_metrics.csv
"""

import argparse
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
import tensorflow as tf

from src.ann import build_ann_model, save_ann_model, train_ann_model
from src.baselines import select_top_k_mutual_information, select_top_k_random_forest
from src.config import AppConfig, load_config
from src.evaluate import evaluate_model_pipeline
from src.fuzzy import FuzzyDecisionLayer, export_all_fuzzy_rules
from src.ga import GeneticFeatureSelector, plot_ga_convergence, plot_ga_convergence_comparison
from src.preprocessing import NSLKDDPreprocessor, process_dataset_and_generate_eda
from src.utils import load_artifact, save_json, set_seeds, setup_logger

logger = setup_logger("run_experiments", log_file=Path("results/run.log"))


def run_experiments_for_seed(
    seed: int,
    config: AppConfig,
    force: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run full feature selection and training pipeline across all 9 ablation models for a single seed."""
    logger.info(f"==================================================")
    logger.info(f"Starting Experiments for SEED {seed} (Mode: {config.mode})")
    logger.info(f"==================================================")
    set_seeds(seed)

    # Step 1: Preprocess data (Strictly fits scalers and fuzzy layer on Train split only)
    datasets, preprocessor = process_dataset_and_generate_eda(config=config, seed=seed)
    X_train_full = datasets["X_train"]
    y_train_full = datasets["y_train"]
    X_val_full = datasets["X_val"]
    y_val_full = datasets["y_val"]
    X_test_full = datasets["X_test"]
    y_test_full = datasets["y_test"]
    class_weights = datasets["class_weights"]
    classes = config.data.classes

    num_original = len(preprocessor.original_feature_names)
    informed_init = getattr(config.ga, "seed_informed_chromosomes", True)

    # Step 2: Standard GA Feature Selection (Fixed mutation/crossover rates)
    logger.info(f"[Seed {seed}] Executing Standard GA Feature Selection...")
    ga_std = GeneticFeatureSelector(
        ga_config=config.ga,
        ann_config=config.ann,
        preprocessor=preprocessor,
        class_weights=class_weights,
        seed=seed,
        use_fuzzy_controller=False,
        seed_informed_chromosomes=informed_init,
    )
    ga_std_chrom, ga_std_fitness, ga_std_df = ga_std.evolve(
        X_train=X_train_full,
        y_train=y_train_full,
        X_val=X_val_full,
        y_val=y_val_full,
    )
    ga_std_csv = config.paths.results_dir / f"ga_history_standard_seed{seed}.csv"
    ga_std_df.to_csv(ga_std_csv, index=False)

    # Step 3: Fuzzy-Adaptive GA Feature Selection (Mamdani dynamic adaptation)
    logger.info(f"[Seed {seed}] Executing Fuzzy-Adaptive GA Feature Selection...")
    ga_adp = GeneticFeatureSelector(
        ga_config=config.ga,
        ann_config=config.ann,
        preprocessor=preprocessor,
        class_weights=class_weights,
        seed=seed,
        use_fuzzy_controller=True,
        seed_informed_chromosomes=informed_init,
    )
    ga_adp_chrom, ga_adp_fitness, ga_adp_df = ga_adp.evolve(
        X_train=X_train_full,
        y_train=y_train_full,
        X_val=X_val_full,
        y_val=y_val_full,
    )
    ga_adp_csv = config.paths.results_dir / f"ga_history_adaptive_seed{seed}.csv"
    ga_adp_df.to_csv(ga_adp_csv, index=False)
    # Also save standard filename for backward compatibility
    ga_adp_df.to_csv(config.paths.results_dir / f"ga_history_seed{seed}.csv", index=False)

    # Save convergence comparison plot
    comp_plot_path = config.paths.plots_dir / f"ga_convergence_comparison_seed{seed}.png"
    plot_ga_convergence_comparison(ga_std_df, ga_adp_df, comp_plot_path, seed=seed)

    k_selected_std = int(np.sum(ga_std_chrom))
    k_selected_adp = int(np.sum(ga_adp_chrom))
    logger.info(
        f"[Seed {seed}] Standard GA selected {k_selected_std}/41 features; Adaptive GA selected {k_selected_adp}/41 features."
    )

    # Step 4: Baseline Feature Selectors (MI and RF) matching Adaptive GA's k
    k_match = k_selected_adp if k_selected_adp > 0 else 15
    logger.info(f"[Seed {seed}] Executing Mutual Information Top-{k_match} Selection...")
    mi_chrom, mi_features, _ = select_top_k_mutual_information(
        X_train=X_train_full,
        y_train=y_train_full,
        preprocessor=preprocessor,
        k=k_match,
        seed=seed,
    )

    logger.info(f"[Seed {seed}] Executing Random Forest Top-{k_match} Selection...")
    rf_chrom, rf_features, _ = select_top_k_random_forest(
        X_train=X_train_full,
        y_train=y_train_full,
        preprocessor=preprocessor,
        k=k_match,
        seed=seed,
    )

    # All Features Chromosome (Baseline)
    all_chrom = np.ones(num_original, dtype=np.int32)

    # Save Feature Selection Masks
    masks_to_save = [
        ("standard_ga", ga_std_chrom, "Standard Genetic Algorithm"),
        ("adaptive_ga", ga_adp_chrom, "Fuzzy-Adaptive Genetic Algorithm"),
        ("ga", ga_adp_chrom, "Genetic Algorithm"),  # compatibility
        ("mi", mi_chrom, "Mutual Information"),
        ("rf", rf_chrom, "Random Forest"),
        ("all", all_chrom, "All Features"),
    ]
    for key, chrom, desc in masks_to_save:
        indices = [int(i) for i, active in enumerate(chrom) if active]
        names = [preprocessor.original_feature_names[i] for i in indices]
        mask_data = {
            "selector": desc,
            "seed": seed,
            "count_selected": len(indices),
            "count_total": num_original,
            "reduction_percentage": round((1.0 - len(indices) / num_original) * 100, 2),
            "selected_feature_indices": indices,
            "selected_feature_names": names,
        }
        save_json(mask_data, config.paths.models_dir / f"selected_mask_{key}_seed{seed}.json")

    # Step 5: Full Retraining & Evaluation for All 9 Ablation Models
    # Structure: (display_name, chromosome, use_fuzzy_input, use_fuzzy_decision, file_prefix)
    models_to_train = [
        ("Baseline ANN", all_chrom, False, False, "baseline_ann"),
        ("Baseline ANN + Fuzzy-decision", all_chrom, False, True, "baseline_ann_fuzzy_decision"),
        ("GA+ANN", ga_std_chrom, False, False, "ga_ann"),
        ("GA(adaptive)+ANN", ga_adp_chrom, False, False, "ga_adaptive_ann"),
        ("GA+Fuzzy-input+ANN", ga_std_chrom, True, False, "ga_fuzzy_input_ann"),
        ("GA(adaptive)+Fuzzy-input+ANN", ga_adp_chrom, True, False, "ga_adaptive_fuzzy_input_ann"),
        (
            "GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision",
            ga_adp_chrom,
            True,
            True,
            "ga_adaptive_fuzzy_input_ann_fuzzy_decision",
        ),
        ("MI+ANN", mi_chrom, False, False, "mi_ann"),
        ("RF+ANN", rf_chrom, False, False, "rf_ann"),
    ]

    seed_results: List[Dict[str, Any]] = []
    seed_decision_stats: Dict[str, Any] = {}
    trained_models_cache: Dict[str, Any] = {}

    raw_attacks_val = datasets["raw_val"][config.data.target_column].astype(str).str.strip().str.lower().values
    raw_attacks_test = datasets["raw_test"][config.data.target_column].astype(str).str.strip().str.lower().values

    for display_name, chrom, use_fuzzy_input, use_fuzzy_decision, file_prefix in models_to_train:
        logger.info(f"--- Training & Evaluating: {display_name} (Seed {seed}) ---")
        n_orig = int(np.sum(chrom))

        susp_band = getattr(config.fuzzy.decision_layer, "suspicious_band", [0.35, 0.65])
        false_alarm_pen = getattr(config.fuzzy.decision_layer, "false_alarm_penalty", 0.30)
        workload_pen = getattr(config.fuzzy.decision_layer, "workload_penalty", 0.10)
        beta_val_penalty = getattr(config.fuzzy.decision_layer, "beta_fpr_penalty", 0.30)

        # Variant 2 isolates decision layer: reuses trained model and probabilities of Variant 1 (Baseline ANN)
        if display_name == "Baseline ANN + Fuzzy-decision":
            cached_model, train_time, X_v, X_te, cached_v_prob, cached_te_prob, n_encoded = trained_models_cache["baseline_ann"]
            model = cached_model
            model_path = config.paths.models_dir / f"model_{file_prefix}_seed{seed}.keras"
            save_ann_model(model, model_path)

            decision_layer = FuzzyDecisionLayer(
                suspicious_band=list(susp_band),
                alert_levels=list(config.fuzzy.decision_layer.alert_levels),
            )

            metrics = evaluate_model_pipeline(
                model=model,
                X_val=X_v,
                y_val=y_val_full,
                X_test=X_te,
                y_test=y_test_full,
                classes=classes,
                model_name=display_name,
                seed=seed,
                cm_dir=config.paths.confusion_matrices_dir,
                training_time=train_time,
                num_orig_features=n_orig,
                num_encoded_features=n_encoded,
                model_path=model_path,
                fuzzy_decision_layer=decision_layer,
                beta_fpr_penalty=beta_val_penalty,
                workload_penalty=workload_pen,
                false_alarm_penalty=false_alarm_pen,
                raw_attacks_val=raw_attacks_val,
                raw_attacks_test=raw_attacks_test,
                reused_val_prob=cached_v_prob,
                reused_test_prob=cached_te_prob,
            )

        # Variant 7 isolates decision layer: reuses trained model and probabilities of Variant 6 (GA(adaptive)+Fuzzy-input+ANN)
        elif display_name == "GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision":
            cached_model, train_time, X_v, X_te, cached_v_prob, cached_te_prob, n_encoded = trained_models_cache["ga_adaptive_fuzzy_input_ann"]
            model = cached_model
            model_path = config.paths.models_dir / f"model_{file_prefix}_seed{seed}.keras"
            save_ann_model(model, model_path)

            decision_layer = FuzzyDecisionLayer(
                suspicious_band=list(susp_band),
                alert_levels=list(config.fuzzy.decision_layer.alert_levels),
            )

            metrics = evaluate_model_pipeline(
                model=model,
                X_val=X_v,
                y_val=y_val_full,
                X_test=X_te,
                y_test=y_test_full,
                classes=classes,
                model_name=display_name,
                seed=seed,
                cm_dir=config.paths.confusion_matrices_dir,
                training_time=train_time,
                num_orig_features=n_orig,
                num_encoded_features=n_encoded,
                model_path=model_path,
                fuzzy_decision_layer=decision_layer,
                beta_fpr_penalty=beta_val_penalty,
                workload_penalty=workload_pen,
                false_alarm_penalty=false_alarm_pen,
                raw_attacks_val=raw_attacks_val,
                raw_attacks_test=raw_attacks_test,
                reused_val_prob=cached_v_prob,
                reused_test_prob=cached_te_prob,
            )

        else:
            # Input transformation: Neuro-Fuzzy (Gaussian memberships) vs Standard Column Slicing
            if use_fuzzy_input:
                input_mode = getattr(config.fuzzy.input_layer, "input_mode", "concat")
                X_tr = preprocessor.transform_neuro_fuzzy(X_train_full, chrom, input_mode=input_mode)
                X_v = preprocessor.transform_neuro_fuzzy(X_val_full, chrom, input_mode=input_mode)
                X_te = preprocessor.transform_neuro_fuzzy(X_test_full, chrom, input_mode=input_mode)
            else:
                col_indices = preprocessor.get_column_indices_for_chromosome(chrom)
                X_tr = X_train_full[:, col_indices]
                X_v = X_val_full[:, col_indices]
                X_te = X_test_full[:, col_indices]

            n_encoded = int(X_tr.shape[1])

            # Build architecture
            model = build_ann_model(
                input_dim=n_encoded,
                config=config.ann,
                num_classes=len(classes),
                name=f"{file_prefix}_seed{seed}",
            )

            # Retrain with identical training budget
            model, history, train_time = train_ann_model(
                model=model,
                X_train=X_tr,
                y_train=y_train_full,
                X_val=X_v,
                y_val=y_val_full,
                config=config.ann,
                class_weights=class_weights,
                verbose=0,
            )

            # Save model
            model_path = config.paths.models_dir / f"model_{file_prefix}_seed{seed}.keras"
            save_ann_model(model, model_path)

            decision_layer = None
            if use_fuzzy_decision:
                decision_layer = FuzzyDecisionLayer(
                    suspicious_band=list(susp_band),
                    alert_levels=list(config.fuzzy.decision_layer.alert_levels),
                )

            metrics = evaluate_model_pipeline(
                model=model,
                X_val=X_v,
                y_val=y_val_full,
                X_test=X_te,
                y_test=y_test_full,
                classes=classes,
                model_name=display_name,
                seed=seed,
                cm_dir=config.paths.confusion_matrices_dir,
                training_time=train_time,
                num_orig_features=n_orig,
                num_encoded_features=n_encoded,
                model_path=model_path,
                fuzzy_decision_layer=decision_layer,
                beta_fpr_penalty=beta_val_penalty,
                workload_penalty=workload_pen,
                false_alarm_penalty=false_alarm_pen,
                raw_attacks_val=raw_attacks_val,
                raw_attacks_test=raw_attacks_test,
            )

            # Cache baseline_ann and ga_adaptive_fuzzy_input_ann models and probabilities for decision layer isolation
            if file_prefix in ["baseline_ann", "ga_adaptive_fuzzy_input_ann"]:
                trained_models_cache[file_prefix] = (
                    model,
                    train_time,
                    X_v,
                    X_te,
                    metrics["val_prob"],
                    metrics["test_prob"],
                    n_encoded,
                )

        seed_results.append(metrics)

        # Log decision stats if model used fuzzy decision layer
        if use_fuzzy_decision and metrics.get("decision_stats"):
            seed_decision_stats[display_name] = {
                "seed": seed,
                "chosen_suspicious_band": metrics["chosen_suspicious_band"],
                "decision_stats": metrics["decision_stats"],
                "suspicious_stats_val": metrics.get("suspicious_stats_val"),
                "suspicious_stats_test": metrics.get("suspicious_stats_test"),
                "alert_metrics_val": metrics.get("alert_metrics_val"),
                "alert_metrics_test": metrics.get("alert_metrics_test"),
            }

        logger.info(
            f"[{display_name}] Val F1: {metrics['val_macro_f1']:.4f} | Val FPR: {metrics['val_false_positive_rate']:.4f} | "
            f"Test F1: {metrics['test_macro_f1']:.4f} | Test FPR: {metrics['test_false_positive_rate']:.4f} | "
            f"Feats: {n_orig}/41 ({n_encoded} enc) | Band: {metrics.get('chosen_suspicious_band')}"
        )

        tf.keras.backend.clear_session()

    return seed_results, seed_decision_stats


def safe_paired_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Execute paired Wilcoxon signed-rank test with paired t-test fallback and power disclaimer."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    n = len(diff)
    mean_diff = float(np.mean(diff))

    if n < 2 or np.all(diff == 0.0):
        return {
            "test_used": "identical_or_single_sample",
            "statistic": None,
            "p_value": 1.0,
            "mean_difference": round(mean_diff, 5),
            "effect_size": 0.0,
            "sample_size": n,
            "is_significant_05": False,
            "power_warning": "Sample size too small (n < 2) or identical pairs; test cannot reject null.",
        }

    try:
        w_res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        total_ranks = n * (n + 1) / 2.0
        r_biserial = float((w_res.statistic - (total_ranks / 2.0)) / max(1.0, (total_ranks / 2.0)))
        p_val = float(w_res.pvalue)
        return {
            "test_used": "paired_wilcoxon_signed_rank",
            "statistic": round(float(w_res.statistic), 5),
            "p_value": round(p_val, 5),
            "mean_difference": round(mean_diff, 5),
            "effect_size_rank_biserial": round(r_biserial, 4),
            "sample_size": n,
            "is_significant_05": bool(p_val < 0.05),
            "power_warning": f"Evaluated across n={n} seeds. Low statistical power; interpret non-significance cautiously.",
        }
    except Exception as e:
        # Fallback to paired t-test
        try:
            t_res = stats.ttest_rel(x, y)
            s_diff = np.std(diff, ddof=1)
            d = float(mean_diff / s_diff) if s_diff > 1e-12 else 0.0
            p_val = float(t_res.pvalue)
            return {
                "test_used": "paired_t_test_fallback",
                "statistic": round(float(t_res.statistic), 5),
                "p_value": round(p_val, 5),
                "mean_difference": round(mean_diff, 5),
                "cohens_d": round(d, 4),
                "sample_size": n,
                "is_significant_05": bool(p_val < 0.05),
                "power_warning": f"Paired t-test used (Wilcoxon failed: {e}). Low statistical power with n={n}.",
            }
        except Exception as e2:
            return {
                "test_used": "error",
                "statistic": None,
                "p_value": None,
                "mean_difference": round(mean_diff, 5),
                "error": f"{e}; {e2}",
                "sample_size": n,
                "is_significant_05": False,
            }


def compute_statistical_tests_all(
    metrics_df: pd.DataFrame,
    full_model: str = "GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision",
    baseline_model: str = "Baseline ANN",
    ga_ann_model: str = "GA+ANN",
) -> Dict[str, Any]:
    """Compute paired statistical tests comparing Full Model vs Baseline ANN and vs GA+ANN."""
    metrics_to_test = [
        ("test_macro_f1", "Macro-F1 Score"),
        ("test_false_positive_rate", "False Positive Rate (FPR)"),
        ("test_accuracy", "Overall Accuracy"),
        ("test_attack_detection_rate", "Attack Detection Rate"),
    ]

    results: Dict[str, Any] = {
        "comparison_vs_baseline": {},
        "comparison_vs_ga_ann": {},
        "metadata": {
            "full_model": full_model,
            "baseline_model": baseline_model,
            "ga_ann_model": ga_ann_model,
            "alpha_significance": 0.05,
        },
    }

    full_runs = metrics_df[metrics_df["model_name"] == full_model].sort_values("seed")
    base_runs = metrics_df[metrics_df["model_name"] == baseline_model].sort_values("seed")
    ga_runs = metrics_df[metrics_df["model_name"] == ga_ann_model].sort_values("seed")

    for col, label in metrics_to_test:
        if not full_runs.empty and not base_runs.empty:
            x = full_runs[col].values
            y = base_runs[col].values
            results["comparison_vs_baseline"][col] = safe_paired_wilcoxon(x, y)
            results["comparison_vs_baseline"][col]["metric_label"] = label

        if not full_runs.empty and not ga_runs.empty:
            x = full_runs[col].values
            y = ga_runs[col].values
            results["comparison_vs_ga_ann"][col] = safe_paired_wilcoxon(x, y)
            results["comparison_vs_ga_ann"][col]["metric_label"] = label

    return results


def generate_win_loss_table(
    summary_dict: Dict[str, Any],
    full_model: str = "GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision",
    baseline_model: str = "Baseline ANN",
    n_seeds: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Programmatically construct win/loss/tie comparison between Full Model and Baseline ANN.

    Rules:
    - If n_seeds < 3, require at least 3 seeds before declaring TIE; output 'INSUFFICIENT DATA (n<3)'.
    - If n_seeds == 1, do not report or use standard deviation.
    - If inference latency difference is smaller than 10%, label as 'WITHIN NOISE' and drop from win/loss.
    """
    full = summary_dict.get(full_model, {})
    base = summary_dict.get(baseline_model, {})

    if n_seeds is None:
        n_seeds = full.get("n_seeds", base.get("n_seeds", 1))

    metrics_spec = [
        ("Macro-F1", "test_macro_f1", "higher", "Primary multi-class balance"),
        ("Accuracy", "test_accuracy", "higher", "Overall classification rate"),
        ("Attack Detection Rate", "test_attack_detection_rate", "higher", "Overall security sensitivity"),
        ("Test False Positive Rate", "test_false_positive_rate", "lower", "False alarms on Normal traffic"),
        ("Val False Positive Rate", "val_false_positive_rate", "lower", "Validation false alarms"),
        ("False Negative Rate", "test_false_negative_rate", "lower", "Missed attack intrusions"),
        ("Original Features", "num_features_original", "lower", "Feature reduction efficiency"),
        ("Encoded Inputs", "num_features_encoded", "lower", "Neural network input width"),
        ("Inference Latency (ms/1k)", "inference_time_per_1000_ms", "lower", "Real-time readiness"),
    ]

    rows = []
    wins, losses, ties, insufficient, noise_count = 0, 0, 0, 0, 0

    full_col_name = "Full_Model" if n_seeds <= 1 else "Full_Model (Mean±Std)"
    base_col_name = "Baseline" if n_seeds <= 1 else "Baseline (Mean±Std)"

    for label, key, direction, desc in metrics_spec:
        full_m = full.get(f"{key}_mean", 0.0)
        full_s = full.get(f"{key}_std", 0.0)
        base_m = base.get(f"{key}_mean", 0.0)
        base_s = base.get(f"{key}_std", 0.0)

        diff = full_m - base_m

        # Latency rule: Drop latency differences smaller than 10% from win/loss, labeling them "WITHIN NOISE"
        if key == "inference_time_per_1000_ms":
            denom = abs(base_m) if abs(base_m) > 1e-9 else 1.0
            rel_diff = abs(diff) / denom
            if rel_diff < 0.10:
                outcome = "WITHIN NOISE"
                noise_count += 1
            else:
                if diff < 0:  # lower latency is better
                    outcome = "WIN"
                    wins += 1
                elif diff > 0:
                    outcome = "LOSS"
                    losses += 1
                else:
                    if n_seeds < 3:
                        outcome = "INSUFFICIENT DATA (n<3)"
                        insufficient += 1
                    else:
                        outcome = "TIE"
                        ties += 1
        else:
            # Tolerance threshold: do not use std when n=1
            is_rate_or_score = any(k in key for k in ["rate", "f1", "accuracy"])
            default_tol = 0.005 if is_rate_or_score else 0.5
            tol = default_tol if n_seeds <= 1 else max(base_s, default_tol)

            if direction == "higher":
                if diff > tol:
                    outcome = "WIN"
                    wins += 1
                elif diff < -tol:
                    outcome = "LOSS"
                    losses += 1
                else:
                    if n_seeds < 3:
                        outcome = "INSUFFICIENT DATA (n<3)"
                        insufficient += 1
                    else:
                        outcome = "TIE"
                        ties += 1
            else:  # lower is better
                if diff < -tol:
                    outcome = "WIN"
                    wins += 1
                elif diff > tol:
                    outcome = "LOSS"
                    losses += 1
                else:
                    if n_seeds < 3:
                        outcome = "INSUFFICIENT DATA (n<3)"
                        insufficient += 1
                    else:
                        outcome = "TIE"
                        ties += 1

        full_val_str = f"{full_m:.4f}" if n_seeds <= 1 else f"{full_m:.4f} ± {full_s:.4f}"
        base_val_str = f"{base_m:.4f}" if n_seeds <= 1 else f"{base_m:.4f} ± {base_s:.4f}"

        rows.append({
            "Metric": label,
            full_col_name: full_val_str,
            base_col_name: base_val_str,
            "Difference": round(diff, 4),
            "Outcome": outcome,
            "Description": desc,
        })

    win_loss_df = pd.DataFrame(rows)
    summary_stats = {
        "full_model": full_model,
        "baseline_model": baseline_model,
        "n_seeds": n_seeds,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "insufficient_data": insufficient,
        "within_noise": noise_count,
        "total_metrics_evaluated": len(rows),
        "win_rate": round(wins / len(rows), 4),
    }
    return win_loss_df, summary_stats


def aggregate_experiment_results(
    all_runs: List[Dict[str, Any]],
    seeds: List[int],
    config: AppConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Aggregate per-run metrics across all 9 ablation models and compute mean/std statistics."""
    records = []
    per_class_records = []

    for r in all_runs:
        row = dict(r)
        per_class = row.pop("per_class_test")
        records.append(row)

        for cls_name, cls_m in per_class.items():
            per_class_records.append({
                "model_name": r["model_name"],
                "seed": r["seed"],
                "class_name": cls_name,
                "precision": cls_m["precision"],
                "recall": cls_m["recall"],
                "f1_score": cls_m["f1_score"],
                "detection_rate": cls_m["detection_rate"],
                "support": cls_m["support"],
            })

    metrics_df = pd.DataFrame(records)
    per_class_df = pd.DataFrame(per_class_records)

    metric_cols = [
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_false_positive_rate",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_weighted_f1",
        "test_false_positive_rate",
        "test_false_negative_rate",
        "test_attack_detection_rate",
        "test_attack_recall_flagged",
        "test_attack_alert_precision",
        "test_normal_flagged_rate",
        "test_analyst_workload_rate",
        "num_features_original",
        "num_features_encoded",
        "training_time_sec",
        "inference_time_per_1000_ms",
        "model_size_kb",
    ]

    summary_dict: Dict[str, Any] = {}
    ablation_rows = []

    for model_name, grp in metrics_df.groupby("model_name", sort=False):
        n_model_seeds = int(grp["seed"].nunique())
        summary_dict[model_name] = {"n_seeds": n_model_seeds}
        ablation_row = {"model_name": model_name, "n_seeds": n_model_seeds}

        for col in metric_cols:
            if col in grp.columns:
                val_mean = round(float(grp[col].dropna().mean()), 5)
                val_std = round(float(grp[col].dropna().std(ddof=0)), 5)
                summary_dict[model_name][f"{col}_mean"] = val_mean
                summary_dict[model_name][f"{col}_std"] = val_std
                ablation_row[f"{col}_mean"] = val_mean
                ablation_row[f"{col}_std"] = val_std

        # Aggregate per-class
        summary_dict[model_name]["per_class"] = {}
        model_pc = per_class_df[per_class_df["model_name"] == model_name]
        for cls_name, c_grp in model_pc.groupby("class_name"):
            summary_dict[model_name]["per_class"][cls_name] = {
                "f1_mean": round(float(c_grp["f1_score"].mean()), 4),
                "f1_std": round(float(c_grp["f1_score"].std(ddof=0)), 4),
                "recall_mean": round(float(c_grp["recall"].mean()), 4),
                "recall_std": round(float(c_grp["recall"].std(ddof=0)), 4),
                "precision_mean": round(float(c_grp["precision"].mean()), 4),
                "precision_std": round(float(c_grp["precision"].std(ddof=0)), 4),
            }

        ablation_rows.append(ablation_row)

    ablation_df = pd.DataFrame(ablation_rows)
    return metrics_df, summary_dict, per_class_df, ablation_df


def compute_baseline_init_noise_estimate(
    config: AppConfig,
    base_split_seed: int = 42,
    init_seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Train the baseline ANN with 5 different weight initialization seeds on the same split.

    Reports mean ± std of test macro-F1 (and other key metrics) to establish the
    experimental noise floor for architectural and feature selection comparisons.
    """
    if init_seeds is None:
        init_seeds = [42, 101, 202, 303, 404]

    logger.info(f"Computing Baseline ANN weight initialization noise floor across {len(init_seeds)} init seeds on split seed {base_split_seed}...")

    # Load split for base_split_seed
    datasets = load_or_create_data(config, seed=base_split_seed)
    (
        X_train_full,
        y_train_full,
        X_val_full,
        y_val_full,
        X_test_full,
        y_test_full,
        preprocessor,
    ) = load_or_create_preprocessor(config, seed=base_split_seed)

    class_weights = compute_class_weights(y_train_full)
    classes = list(config.data.classes)
    num_original = len(preprocessor.original_feature_names)
    all_chrom = np.ones(num_original, dtype=bool)
    col_indices = preprocessor.get_column_indices_for_chromosome(all_chrom)

    X_tr = X_train_full[:, col_indices]
    X_v = X_val_full[:, col_indices]
    X_te = X_test_full[:, col_indices]
    input_dim = int(X_tr.shape[1])

    runs = []
    for init_seed in init_seeds:
        logger.info(f"  Training Baseline ANN with init seed {init_seed}...")
        np.random.seed(init_seed)
        tf.random.set_seed(init_seed)

        model = build_ann_model(
            input_dim=input_dim,
            config=config.ann,
            num_classes=len(classes),
            name=f"baseline_noise_init{init_seed}",
        )

        model, history, train_time = train_ann_model(
            model=model,
            X_train=X_tr,
            y_train=y_train_full,
            X_val=X_v,
            y_val=y_val_full,
            config=config.ann,
            class_weights=class_weights,
            verbose=0,
        )

        val_pred, val_prob = predict_ann(model, X_v)
        test_pred, test_prob = predict_ann(model, X_te)

        val_m = compute_comprehensive_metrics(y_val_full, val_pred, val_prob, classes, split_name="validation")
        test_m = compute_comprehensive_metrics(y_test_full, test_pred, test_prob, classes, split_name="test")

        runs.append({
            "init_seed": init_seed,
            "test_macro_f1": test_m["macro_f1"],
            "test_accuracy": test_m["accuracy"],
            "test_false_positive_rate": test_m["false_positive_rate"],
            "test_attack_detection_rate": test_m["overall_attack_detection_rate"],
            "val_macro_f1": val_m["macro_f1"],
            "val_false_positive_rate": val_m["false_positive_rate"],
            "training_time_sec": round(train_time, 2),
        })

        tf.keras.backend.clear_session()

    f1_vals = [r["test_macro_f1"] for r in runs]
    acc_vals = [r["test_accuracy"] for r in runs]
    fpr_vals = [r["test_false_positive_rate"] for r in runs]
    adr_vals = [r["test_attack_detection_rate"] for r in runs]

    f1_mean = float(np.mean(f1_vals))
    f1_std = float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0

    noise_report = {
        "base_split_seed": base_split_seed,
        "n_init_seeds": len(init_seeds),
        "init_seeds": init_seeds,
        "runs": runs,
        "statistics": {
            "test_macro_f1_mean": round(f1_mean, 5),
            "test_macro_f1_std": round(f1_std, 5),
            "test_macro_f1_min": round(float(np.min(f1_vals)), 5),
            "test_macro_f1_max": round(float(np.max(f1_vals)), 5),
            "test_macro_f1_range": round(float(np.max(f1_vals) - np.min(f1_vals)), 5),
            "test_accuracy_mean": round(float(np.mean(acc_vals)), 5),
            "test_accuracy_std": round(float(np.std(acc_vals, ddof=1)), 5),
            "test_fpr_mean": round(float(np.mean(fpr_vals)), 5),
            "test_fpr_std": round(float(np.std(fpr_vals, ddof=1)), 5),
            "test_attack_detection_rate_mean": round(float(np.mean(adr_vals)), 5),
            "test_attack_detection_rate_std": round(float(np.std(adr_vals, ddof=1)), 5),
        },
        "summary": f"Baseline ANN Test Macro-F1: {f1_mean:.4f} ± {f1_std:.4f} across {len(init_seeds)} random initialization seeds on identical split.",
    }

    out_path = config.paths.results_dir / "baseline_noise_estimate.json"
    save_json(noise_report, out_path)
    logger.info(f"Saved baseline noise estimate to {out_path}: {noise_report['summary']}")
    return noise_report


def run_all(
    mode: str = "fast",
    seeds_arg: Optional[str] = None,
    force: bool = False,
) -> None:
    """Main execution orchestrator for the complete ablation suite."""
    start_time = datetime.now()

    # Determine seeds
    seed_list: Optional[List[int]] = None
    if seeds_arg:
        if seeds_arg.isdigit():
            n = int(seeds_arg)
            if n <= 1:
                seed_list = [42]
            elif n == 2:
                seed_list = [42, 123]
            elif n == 3:
                seed_list = [42, 123, 999]
            else:
                seed_list = [42, 123, 999, 2024, 777][:n]
        elif "," in seeds_arg:
            seed_list = [int(s.strip()) for s in seeds_arg.split(",")]

    config = load_config(mode=mode, seeds=seed_list)
    logger.info(f"Loaded configuration for project: {config.project_name}")
    logger.info(f"Execution Mode: {mode}, Seeds: {config.seeds}")

    # Ensure rule book is exported
    export_all_fuzzy_rules(config.paths.results_dir / "fuzzy_rules.json")

    all_seed_results: List[Dict[str, Any]] = []
    all_decision_stats: Dict[str, Any] = {}

    for seed in config.seeds:
        seed_results, seed_decision = run_experiments_for_seed(seed=seed, config=config, force=force)
        all_seed_results.extend(seed_results)
        all_decision_stats[str(seed)] = seed_decision

    # Aggregate results across seeds
    logger.info("Aggregating metrics across all 9 ablation models...")
    metrics_df, summary_dict, per_class_df, ablation_df = aggregate_experiment_results(
        all_runs=all_seed_results,
        seeds=config.seeds,
        config=config,
    )

    # Save ablation_summary.csv
    ablation_csv_path = config.paths.results_dir / "ablation_summary.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)
    logger.info(f"Saved complete ablation summary to {ablation_csv_path}")

    # Save metrics CSV and summary JSON
    metrics_csv_path = config.paths.results_dir / "metrics_all_runs.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved all run metrics to {metrics_csv_path}")

    per_class_csv_path = config.paths.results_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    logger.info(f"Saved per-class metrics to {per_class_csv_path}")

    summary_json_path = config.paths.results_dir / "metrics_summary.json"
    save_json(summary_dict, summary_json_path)
    logger.info(f"Saved metrics summary to {summary_json_path}")

    # Save fuzzy decision statistics
    decision_stats_path = config.paths.results_dir / "fuzzy_decision_stats.json"
    save_json(all_decision_stats, decision_stats_path)
    logger.info(f"Saved fuzzy decision stats to {decision_stats_path}")

    # Statistical Tests
    stat_tests = compute_statistical_tests_all(
        metrics_df=metrics_df,
        full_model="GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision",
        baseline_model="Baseline ANN",
        ga_ann_model="GA+ANN",
    )
    stat_path = config.paths.results_dir / "statistical_tests.json"
    save_json(stat_tests, stat_path)
    logger.info(f"Saved statistical tests to {stat_path}")

    # Win/Loss Table
    win_loss_df, win_loss_stats = generate_win_loss_table(
        summary_dict=summary_dict,
        full_model="GA(adaptive)+Fuzzy-input+ANN+Fuzzy-decision",
        baseline_model="Baseline ANN",
        n_seeds=len(config.seeds),
    )
    win_loss_csv_path = config.paths.results_dir / "win_loss_table.csv"
    win_loss_df.to_csv(win_loss_csv_path, index=False)
    save_json(win_loss_stats, config.paths.results_dir / "win_loss_table.json")
    # Baseline ANN Weight Initialization Noise Estimate (5 random init seeds on same split)
    logger.info("Computing Baseline ANN weight initialization noise floor...")
    noise_estimate = compute_baseline_init_noise_estimate(
        config=config,
        base_split_seed=config.seeds[0],
        init_seeds=[42, 101, 202, 303, 404],
    )

    # Save run metadata
    metadata = {
        "project": config.project_name,
        "mode": mode,
        "seeds": config.seeds,
        "num_models_evaluated": len(ablation_df),
        "models_evaluated": ablation_df["model_name"].tolist(),
        "start_time": start_time.isoformat(),
        "completion_time": datetime.now().isoformat(),
        "total_elapsed_sec": round((datetime.now() - start_time).total_seconds(), 2),
        "library_versions": {
            "tensorflow": tf.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "config": {
            "val_split_ratio": config.data.val_split_ratio,
            "classes": config.data.classes,
            "ann": {
                "hidden_layers": config.ann.hidden_layers,
                "epochs": config.ann.epochs,
                "batch_size": config.ann.batch_size,
            },
            "ga": {
                "population_size": config.ga.population_size,
                "generations": config.ga.generations,
                "alpha_penalty": config.ga.alpha_penalty,
                "beta_penalty": config.ga.beta_penalty,
            },
            "fuzzy": {
                "decision_layer_beta_fpr_penalty": config.fuzzy.decision_layer.beta_fpr_penalty,
                "suspicious_band": config.fuzzy.decision_layer.suspicious_band,
            },
        },
        "baseline_noise_estimate": noise_estimate["statistics"],
    }
    save_json(metadata, config.paths.results_dir / "run_metadata.json")

    # Automatically regenerate all figures (both legacy and neuro-fuzzy publication figures)
    logger.info("Regenerating all diagnostic figures and publication plots...")
    try:
        from src.plots import generate_all_plots
        from src.fuzzy_plots import generate_all_fuzzy_plots
        plots_legacy = generate_all_plots(config)
        plots_fuzzy = generate_all_fuzzy_plots(config)
        logger.info(f"Rebuilt {len(plots_legacy)} legacy figures and {len(plots_fuzzy)} publication figures in {config.paths.plots_dir}")
    except Exception as e:
        logger.warning(f"Plot regeneration encountered warning: {e}")

    logger.info("==================================================")
    logger.info("EXPERIMENT RUN COMPLETE!")
    logger.info(f"Evaluated {len(ablation_df)} models across {len(config.seeds)} seed(s).")
    logger.info(f"Saved ablation summary to {ablation_csv_path}")
    logger.info(f"Saved win/loss table to {win_loss_csv_path}")
    logger.info(f"Saved statistical tests to {stat_path}")
    logger.info("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Run Evolutionary Feature Selection Experiments on NSL-KDD")
    parser.add_argument("--mode", type=str, default="fast", choices=["smoke", "fast", "full"], help="Execution mode")
    parser.add_argument("--seeds", type=str, default="1", help="Number of seeds or comma-separated list of seeds")
    parser.add_argument("--force", action="store_true", help="Force re-run even if results exist")
    args = parser.parse_args()

    run_all(mode=args.mode, seeds_arg=args.seeds, force=args.force)


if __name__ == "__main__":
    main()
