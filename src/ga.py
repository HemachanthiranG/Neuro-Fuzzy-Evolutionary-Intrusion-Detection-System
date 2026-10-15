"""Pure Python/NumPy Genetic Algorithm for Feature Selection in NSL-KDD.

Includes:
1. Standard Genetic Algorithm for 41-feature subset selection.
2. Fuzzy-Adaptive GA Controller (Mamdani FLC) dynamically adapting pm and pc.
3. Enhanced fitness function: Macro-F1 - alpha * (n/41) - beta * FPR(Normal vs Attack).
4. Informed initial population seeding (Mutual Information & Random Forest).
5. Comprehensive history logging and publication-quality convergence comparison plotting.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, optimizers

from src.ann import build_ann_model
from src.baselines import select_top_k_mutual_information, select_top_k_random_forest
from src.config import ANNConfig, AppConfig, GAConfig, load_config
from src.fuzzy import FuzzyGAController
from src.preprocessing import NSLKDDPreprocessor, process_dataset_and_generate_eda
from src.utils import save_json, set_seeds, setup_logger

logger = setup_logger("ga")


class GeneticFeatureSelector:
    """Genetic Algorithm for original feature subset selection on NSL-KDD.

    Guarantees:
    - Pure Python and NumPy implementation.
    - Chromosome represents 41 original features (1-to-many column mapping).
    - Enforces >= 1 feature constraint via chromosome repair.
    - Operates exclusively on training split; evaluated against validation split.
    - Supports dynamic parameter adaptation via Mamdani Fuzzy Logic Controller.
    - Enhanced fitness penalizing both feature redundancy (alpha) and false positive rate (beta).
    - Caches evaluations to eliminate redundant ANN training.
    """

    def __init__(
        self,
        ga_config: GAConfig,
        ann_config: ANNConfig,
        preprocessor: NSLKDDPreprocessor,
        class_weights: Optional[Dict[int, float]] = None,
        seed: int = 42,
        use_fuzzy_controller: Optional[bool] = None,
        seed_informed_chromosomes: Optional[bool] = None,
        fuzzy_controller: Optional[FuzzyGAController] = None,
    ):
        self.ga_cfg = ga_config
        self.ann_cfg = ann_config
        self.preprocessor = preprocessor
        self.class_weights = class_weights
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.num_original_features = len(preprocessor.original_feature_names)
        self.fitness_cache: Dict[Tuple[int, ...], Tuple[float, float, float, int]] = {}

        # Controller settings
        self.use_fuzzy_controller = (
            use_fuzzy_controller
            if use_fuzzy_controller is not None
            else getattr(self.ga_cfg, "use_fuzzy_controller", False)
        )
        self.seed_informed_chromosomes = (
            seed_informed_chromosomes
            if seed_informed_chromosomes is not None
            else getattr(self.ga_cfg, "seed_informed_chromosomes", False)
        )

        if self.use_fuzzy_controller:
            self.fuzzy_controller = (
                fuzzy_controller
                if fuzzy_controller is not None
                else FuzzyGAController(
                    min_mutation_rate=0.005,
                    max_mutation_rate=0.10,
                    min_crossover_rate=0.60,
                    max_crossover_rate=0.95,
                    max_stagnation_patience=max(1, self.ga_cfg.early_stopping_generations),
                )
            )
        else:
            self.fuzzy_controller = None

    def repair_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """Ensure at least one feature is active in the chromosome."""
        chrom = chromosome.copy()
        if np.sum(chrom) == 0:
            rand_idx = self.rng.randint(0, self.num_original_features)
            chrom[rand_idx] = 1
        return chrom

    def initialize_population(
        self,
        X_train_seed: Optional[np.ndarray] = None,
        y_train_seed: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Initialize population with varied sparsity and optional informed chromosomes."""
        pop = []
        seeded_count = 0

        # Optional informed chromosome seeding using training split only
        if self.seed_informed_chromosomes and X_train_seed is not None and y_train_seed is not None:
            try:
                # Seed with top-15 MI features
                mi_chrom, _, _ = select_top_k_mutual_information(
                    X_train=X_train_seed,
                    y_train=y_train_seed,
                    preprocessor=self.preprocessor,
                    k=min(15, self.num_original_features),
                    seed=self.seed,
                    subsample_size=min(5000, len(X_train_seed)),
                )
                pop.append(self.repair_chromosome(mi_chrom))
                seeded_count += 1

                # Seed with top-15 RF features
                rf_chrom, _, _ = select_top_k_random_forest(
                    X_train=X_train_seed,
                    y_train=y_train_seed,
                    preprocessor=self.preprocessor,
                    k=min(15, self.num_original_features),
                    seed=self.seed,
                    subsample_size=min(5000, len(X_train_seed)),
                )
                pop.append(self.repair_chromosome(rf_chrom))
                seeded_count += 1
                logger.info(
                    f"Seeded GA initial population with {seeded_count} informed chromosomes (MI and RF)."
                )
            except Exception as e:
                logger.warning(
                    f"Informed seeding encountered error: {e}. Falling back to random initialization."
                )

        # Fill remaining population with uniform feature probability sampling
        while len(pop) < self.ga_cfg.population_size:
            p = self.rng.uniform(0.2, 0.6)
            individual = self.rng.binomial(1, p, size=self.num_original_features).astype(np.int32)
            pop.append(self.repair_chromosome(individual))

        return pop

    def evaluate_chromosome(
        self,
        chromosome: np.ndarray,
        X_train_sub: np.ndarray,
        y_train_sub: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, float, float, int]:
        """Evaluate chromosome fitness using a fast ANN on validation macro-F1 and FPR.

        Fitness = validation_macro_f1 - alpha * (n_selected / 41) - beta * FPR(Normal vs Attack)
        Returns: (fitness, raw_macro_f1, fpr, n_selected)
        """
        chrom_key = tuple(chromosome.tolist())
        if chrom_key in self.fitness_cache:
            return self.fitness_cache[chrom_key]

        col_indices = self.preprocessor.get_column_indices_for_chromosome(chromosome)
        n_selected = int(np.sum(chromosome))

        X_tr = X_train_sub[:, col_indices]
        X_v = X_val[:, col_indices]

        # Fast ANN training configuration
        fast_ann_cfg = ANNConfig(
            hidden_layers=[64, 32],
            dropouts=[0.2, 0.1],
            activations=["relu", "relu"],
            learning_rate=self.ann_cfg.learning_rate * 1.5,
            batch_size=self.ga_cfg.fast_ann_batch_size,
            epochs=self.ga_cfg.fast_ann_epochs,
            use_batch_norm=False,
            early_stopping_patience=2,
            reduce_lr_patience=1,
            reduce_lr_factor=0.5,
            min_lr=1e-5,
            use_class_weights=self.ann_cfg.use_class_weights,
        )

        model = build_ann_model(
            input_dim=len(col_indices),
            config=fast_ann_cfg,
            num_classes=len(self.preprocessor.classes),
            name="fast_ann_eval",
        )

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True, verbose=0
            )
        ]

        fit_weights = self.class_weights if self.ann_cfg.use_class_weights else None
        model.fit(
            X_tr,
            y_train_sub,
            validation_data=(X_v, y_val),
            epochs=self.ga_cfg.fast_ann_epochs,
            batch_size=self.ga_cfg.fast_ann_batch_size,
            callbacks=cb_list,
            class_weight=fit_weights,
            verbose=0,
        )

        # Validation prediction
        probs = model.predict(X_v, batch_size=512, verbose=0)
        preds = np.argmax(probs, axis=1)
        raw_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        # False Positive Rate: Normal is class 0, Attacks are classes 1..4
        normal_mask = (y_val == 0)
        normal_count = np.sum(normal_mask)
        if normal_count > 0:
            fp = np.sum(normal_mask & (preds != 0))
            fpr = float(fp / normal_count)
        else:
            fpr = 0.0

        # Multi-objective penalty
        alpha_penalty = self.ga_cfg.alpha_penalty * (n_selected / float(self.num_original_features))
        beta_penalty = getattr(self.ga_cfg, "beta_penalty", 0.05) * fpr
        fitness = raw_f1 - alpha_penalty - beta_penalty

        # Clear session to release memory
        tf.keras.backend.clear_session()

        result = (fitness, raw_f1, fpr, n_selected)
        self.fitness_cache[chrom_key] = result
        return result

    def tournament_selection(
        self, population: List[np.ndarray], fitnesses: List[float]
    ) -> np.ndarray:
        """Select one individual via tournament selection."""
        k = min(self.ga_cfg.tournament_size, len(population))
        selected_indices = self.rng.choice(len(population), size=k, replace=False)
        best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        pc: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover with configured or adapted crossover rate."""
        rate = pc if pc is not None else self.ga_cfg.crossover_rate
        if self.rng.rand() < rate:
            mask = self.rng.binomial(1, 0.5, size=self.num_original_features).astype(bool)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(
        self,
        chromosome: np.ndarray,
        pm: Optional[float] = None,
    ) -> np.ndarray:
        """Bit-flip mutation with configured or adapted mutation rate."""
        rate = pm if pm is not None else self.ga_cfg.mutation_rate
        mutated = chromosome.copy()
        mutation_mask = self.rng.rand(self.num_original_features) < rate
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        return self.repair_chromosome(mutated)

    def evolve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """Execute genetic search and return best chromosome, fitness, and history DataFrame."""
        mode_str = "Adaptive Fuzzy GA" if self.use_fuzzy_controller else "Standard GA"
        logger.info(
            f"[Seed {self.seed}] Starting {mode_str}: Pop={self.ga_cfg.population_size}, Gen={self.ga_cfg.generations}, Subsample={self.ga_cfg.eval_subsample_ratio:.0%}"
        )

        # Create stratified subsample of training data for fast fitness evaluations
        subsample_ratio = self.ga_cfg.eval_subsample_ratio
        if subsample_ratio < 1.0:
            class_counts = np.bincount(y_train)
            stratify_target = y_train if len(class_counts) > 0 and np.min(class_counts) >= 2 else None
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train,
                y_train,
                train_size=subsample_ratio,
                stratify=stratify_target,
                random_state=self.seed,
            )
        else:
            X_train_sub, y_train_sub = X_train, y_train

        logger.info(f"GA fast evaluation set: {len(X_train_sub)} training samples.")

        population = self.initialize_population(X_train_sub, y_train_sub)
        history_records = []

        best_overall_fitness = -float("inf")
        best_overall_chromosome = population[0].copy()
        best_overall_f1 = 0.0
        best_overall_fpr = 0.0
        best_overall_features = self.num_original_features
        no_improvement_count = 0

        for gen in range(self.ga_cfg.generations):
            gen_start_time = time.perf_counter()

            # Evaluate population
            eval_results = [
                self.evaluate_chromosome(ind, X_train_sub, y_train_sub, X_val, y_val)
                for ind in population
            ]

            fitnesses = [res[0] for res in eval_results]
            raw_f1s = [res[1] for res in eval_results]
            fprs = [res[2] for res in eval_results]
            feature_counts = [res[3] for res in eval_results]

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_f1 = raw_f1s[gen_best_idx]
            gen_best_fpr = fprs[gen_best_idx]
            gen_best_features = feature_counts[gen_best_idx]
            gen_mean_fitness = float(np.mean(fitnesses))
            gen_worst_fitness = float(np.min(fitnesses))

            # Compute population metrics
            diversity = FuzzyGAController.compute_population_diversity(population)
            stagnation = (
                self.fuzzy_controller.compute_stagnation(no_improvement_count)
                if self.fuzzy_controller is not None
                else min(
                    1.0,
                    float(no_improvement_count) / max(1, self.ga_cfg.early_stopping_generations),
                )
            )

            # Adapt parameters or use fixed rates
            if self.use_fuzzy_controller and self.fuzzy_controller is not None:
                current_pm, current_pc, _ = self.fuzzy_controller.evaluate(diversity, stagnation)
            else:
                current_pm = self.ga_cfg.mutation_rate
                current_pc = self.ga_cfg.crossover_rate

            gen_time = time.perf_counter() - gen_start_time

            # Check overall best
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_chromosome = population[gen_best_idx].copy()
                best_overall_f1 = gen_best_f1
                best_overall_fpr = gen_best_fpr
                best_overall_features = gen_best_features
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            record = {
                "generation": gen,
                "best_fitness": round(gen_best_fitness, 5),
                "mean_fitness": round(gen_mean_fitness, 5),
                "worst_fitness": round(gen_worst_fitness, 5),
                "best_macro_f1": round(gen_best_f1, 5),
                "best_fpr": round(gen_best_fpr, 5),
                "best_features_count": gen_best_features,
                "overall_best_fitness": round(best_overall_fitness, 5),
                "overall_best_features": best_overall_features,
                "diversity": round(diversity, 4),
                "stagnation": round(stagnation, 4),
                "mutation_rate": round(current_pm, 5),
                "crossover_rate": round(current_pc, 4),
                "generation_time_sec": round(gen_time, 2),
                "cache_size": len(self.fitness_cache),
            }
            history_records.append(record)

            logger.info(
                f"[Gen {gen:02d}] Best: {gen_best_fitness:.4f} (F1: {gen_best_f1:.4f}, FPR: {gen_best_fpr:.4f}, Feats: {gen_best_features}/41) | "
                f"div: {diversity:.3f}, stag: {stagnation:.2f} -> pm: {current_pm:.4f}, pc: {current_pc:.2f} | Time: {gen_time:.1f}s"
            )

            # Early stopping check
            if no_improvement_count >= self.ga_cfg.early_stopping_generations:
                logger.info(
                    f"Early stopping triggered at generation {gen} after {no_improvement_count} generations without improvement."
                )
                break

            # Create next generation
            # Elitism: retain top N individuals
            elite_indices = np.argsort(fitnesses)[::-1][: self.ga_cfg.elitism_count]
            next_generation = [population[idx].copy() for idx in elite_indices]

            while len(next_generation) < self.ga_cfg.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2, pc=current_pc)
                child1 = self.mutate(child1, pm=current_pm)
                child2 = self.mutate(child2, pm=current_pm)
                next_generation.append(child1)
                if len(next_generation) < self.ga_cfg.population_size:
                    next_generation.append(child2)

            population = next_generation

        history_df = pd.DataFrame(history_records)
        logger.info(
            f"{mode_str} Search complete: Best Fitness = {best_overall_fitness:.4f}, Best Macro-F1 = {best_overall_f1:.4f}, Selected Features = {best_overall_features}/41."
        )
        return best_overall_chromosome, best_overall_fitness, history_df


def plot_ga_convergence(history_df: pd.DataFrame, output_path: Path, title_suffix: str = "") -> None:
    """Generate publication-quality GA convergence plot."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    gens = history_df["generation"]

    # Fitness curves
    ax1.plot(gens, history_df["best_fitness"], "o-", color="#1f77b4", linewidth=2, label="Best Fitness")
    ax1.plot(gens, history_df["mean_fitness"], "s--", color="#2ca02c", linewidth=1.5, label="Mean Fitness")
    ax1.plot(gens, history_df["worst_fitness"], "^:", color="#d62728", linewidth=1, label="Worst Fitness")

    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("Fitness (Macro-F1 - Penalty)", fontsize=11, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title(f"Genetic Algorithm Feature Selection Convergence {title_suffix}".strip(), fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Feature count on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(gens, history_df["best_features_count"], "D-.", color="#ff7f0e", linewidth=1.5, alpha=0.85, label="# Features (Best)")
    ax2.set_ylabel("Selected Features Count (out of 41)", fontsize=11, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim(0, 45)
    ax2.grid(False)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved GA convergence plot to {output_path}")


def plot_ga_convergence_comparison(
    standard_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    output_path: Path,
    seed: int = 42,
) -> None:
    """Generate publication-quality comparative plot between Standard GA and Fuzzy-Adaptive GA."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=300, sharex=True)

    gens_std = standard_df["generation"]
    gens_adp = adaptive_df["generation"]

    # --- Top Subplot: Fitness and Feature Count Comparison ---
    ax1.plot(
        gens_std, standard_df["best_fitness"], "o--", color="#1f77b4", linewidth=2, label="Standard GA Best Fitness"
    )
    ax1.plot(
        gens_adp, adaptive_df["best_fitness"], "s-", color="#d62728", linewidth=2.5, label="Fuzzy-Adaptive GA Best Fitness"
    )
    ax1.plot(
        gens_std, standard_df["mean_fitness"], ":", color="#1f77b4", alpha=0.6, label="Standard GA Mean"
    )
    ax1.plot(
        gens_adp, adaptive_df["mean_fitness"], "-.", color="#d62728", alpha=0.6, label="Fuzzy-Adaptive Mean"
    )

    ax1.set_ylabel("Fitness (Macro-F1 - Penalties)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Convergence Comparison: Standard GA vs. Fuzzy-Adaptive GA (Seed {seed})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis on top plot for feature counts
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        gens_std, standard_df["best_features_count"], "v--", color="#2ca02c", alpha=0.7, label="Standard Features"
    )
    ax1_twin.plot(
        gens_adp, adaptive_df["best_features_count"], "^-", color="#9467bd", alpha=0.85, label="Fuzzy-Adaptive Features"
    )
    ax1_twin.set_ylabel("Selected Features (41 max)", fontsize=11, color="#333333")
    ax1_twin.set_ylim(0, 45)
    ax1_twin.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_tw, labels1_tw = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_tw, labels1 + labels1_tw, loc="lower right", framealpha=0.9, fontsize=9)

    # --- Bottom Subplot: Dynamic Parameter Adaptation ---
    ax2.plot(
        gens_adp, adaptive_df["mutation_rate"], "o-", color="#e377c2", linewidth=2, label="Adaptive Mutation Rate ($p_m$)"
    )
    if "mutation_rate" in standard_df.columns:
        ax2.axhline(
            y=standard_df["mutation_rate"].iloc[0],
            color="#e377c2",
            linestyle="--",
            alpha=0.6,
            label="Fixed Standard $p_m$",
        )

    ax2.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mutation Rate ($p_m$)", fontsize=11, color="#e377c2", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#e377c2")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis on bottom plot for Crossover Rate and Diversity
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        gens_adp, adaptive_df["crossover_rate"], "s-", color="#17becf", linewidth=2, label="Adaptive Crossover Rate ($p_c$)"
    )
    if "crossover_rate" in standard_df.columns:
        ax2_twin.axhline(
            y=standard_df["crossover_rate"].iloc[0],
            color="#17becf",
            linestyle="--",
            alpha=0.6,
            label="Fixed Standard $p_c$",
        )
    ax2_twin.plot(
        gens_adp, adaptive_df["diversity"], "D:", color="#8c564b", alpha=0.7, label="Population Diversity"
    )
    ax2_twin.set_ylabel("Crossover Rate ($p_c$) / Diversity", fontsize=11, color="#17becf", fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#17becf")
    ax2_twin.set_ylim(0.0, 1.05)
    ax2_twin.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_tw, labels2_tw = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines2 + lines2_tw, labels2 + labels2_tw, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved GA convergence comparison plot to {output_path}")


def run_ga_single_seed(
    mode: str = "fast",
    seed: int = 42,
    config: Optional[AppConfig] = None,
    use_fuzzy_controller: bool = False,
    seed_informed_chromosomes: Optional[bool] = None,
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """Execute GA feature selection for a single seed and save outputs."""
    set_seeds(seed)
    if config is None:
        config = load_config(mode=mode, seeds=[seed])

    if seed_informed_chromosomes is None:
        seed_informed_chromosomes = getattr(config.ga, "seed_informed_chromosomes", True)

    # Load data
    datasets, preprocessor = process_dataset_and_generate_eda(config=config, seed=seed)

    ga = GeneticFeatureSelector(
        ga_config=config.ga,
        ann_config=config.ann,
        preprocessor=preprocessor,
        class_weights=datasets["class_weights"],
        seed=seed,
        use_fuzzy_controller=use_fuzzy_controller,
        seed_informed_chromosomes=seed_informed_chromosomes,
    )

    best_chrom, best_fitness, history_df = ga.evolve(
        X_train=datasets["X_train"],
        y_train=datasets["y_train"],
        X_val=datasets["X_val"],
        y_val=datasets["y_val"],
    )

    # Save history CSV
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    tag = "adaptive" if use_fuzzy_controller else "standard"
    history_csv_path = config.paths.results_dir / f"ga_history_{tag}_seed{seed}.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Saved GA history to {history_csv_path}")

    # Also save standard filename for backward compatibility
    if not use_fuzzy_controller:
        history_df.to_csv(config.paths.results_dir / f"ga_history_seed{seed}.csv", index=False)

    # Plot convergence
    config.paths.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = config.paths.plots_dir / f"ga_convergence_{tag}_seed{seed}.png"
    plot_ga_convergence(history_df, plot_path, title_suffix=f"({tag.capitalize()}, Seed {seed})")

    # Save selected feature mask
    selected_indices = [int(i) for i, active in enumerate(best_chrom) if active]
    selected_names = [preprocessor.original_feature_names[i] for i in selected_indices]
    mask_info = {
        "seed": seed,
        "variant": tag,
        "best_fitness": round(float(best_fitness), 5),
        "selected_original_feature_indices": selected_indices,
        "selected_original_feature_names": selected_names,
        "count_selected": len(selected_indices),
        "count_total": len(best_chrom),
        "reduction_percentage": round((1.0 - len(selected_indices) / len(best_chrom)) * 100, 2),
    }

    mask_path = config.paths.models_dir / f"selected_mask_{tag}_seed{seed}.json"
    save_json(mask_info, mask_path)
    if not use_fuzzy_controller:
        save_json(mask_info, config.paths.models_dir / f"selected_mask_seed{seed}.json")
    logger.info(f"Saved selected feature mask to {mask_path}")

    return best_chrom, best_fitness, history_df


def run_ga_comparison(
    mode: str = "fast",
    seed: int = 42,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Execute both Standard GA and Fuzzy-Adaptive GA on the same seed and compare."""
    logger.info(f"=== Starting GA Convergence Comparison (Mode={mode}, Seed={seed}) ===")
    if config is None:
        config = load_config(mode=mode, seeds=[seed])

    informed_init = getattr(config.ga, "seed_informed_chromosomes", True)
    logger.info(f"GA initialization policy for both variants: informed={informed_init}")

    # 1. Run Standard GA
    logger.info("--- Running Standard GA ---")
    std_chrom, std_fit, std_df = run_ga_single_seed(
        mode=mode,
        seed=seed,
        config=config,
        use_fuzzy_controller=False,
        seed_informed_chromosomes=informed_init,
    )

    # 2. Run Fuzzy-Adaptive GA
    logger.info("--- Running Fuzzy-Adaptive GA ---")
    adp_chrom, adp_fit, adp_df = run_ga_single_seed(
        mode=mode,
        seed=seed,
        config=config,
        use_fuzzy_controller=True,
        seed_informed_chromosomes=informed_init,
    )

    # 3. Generate Comparative Plot
    comp_plot_path = config.paths.plots_dir / f"ga_convergence_comparison_seed{seed}.png"
    plot_ga_convergence_comparison(std_df, adp_df, comp_plot_path, seed=seed)

    # 4. Save fuzzy rules definitions
    controller = FuzzyGAController()
    rules_dict = {
        "mamdani_ga_controller": {
            "num_rules": len(controller.rules),
            "inputs": {
                "diversity": "Normalized mean pairwise Hamming distance [0, 1]",
                "stagnation": "Normalized generations without fitness improvement [0, 1]",
            },
            "outputs": {
                "mutation_rate": f"Adapted mutation rate [{controller.min_pm}, {controller.max_pm}]",
                "crossover_rate": f"Adapted crossover rate [{controller.min_pc}, {controller.max_pc}]",
            },
            "rules": controller.export_rules_dict(),
        }
    }
    rules_path = config.paths.results_dir / "fuzzy_rules.json"
    save_json(rules_dict, rules_path)
    logger.info(f"Saved fuzzy rules to {rules_path}")

    summary = {
        "seed": seed,
        "standard_ga": {
            "best_fitness": round(float(std_fit), 5),
            "features_selected": int(np.sum(std_chrom)),
            "generations_run": len(std_df),
        },
        "fuzzy_adaptive_ga": {
            "best_fitness": round(float(adp_fit), 5),
            "features_selected": int(np.sum(adp_chrom)),
            "generations_run": len(adp_df),
        },
        "comparison_plot": str(comp_plot_path),
    }
    return summary


if __name__ == "__main__":
    summary = run_ga_comparison(mode="fast", seed=42)
    print("GA Comparison Complete!")
    print(summary)
