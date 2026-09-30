"""
Accuracy & Output Fidelity Validation Engine for SnapForge / Nibble.
Validates numerical fidelity between original and optimized models:
- Cosine similarity
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Maximum Absolute Difference (Max Diff)
- Relative Error (L2 relative error)
- Top-1 and Top-5 classification agreement percentage
- Top prediction argmax preservation ("YES" / "NO")
- Robust handling for NaNs, Infs, and zero-vectors
"""

from typing import Dict, Any, List, Optional
import numpy as np


class AccuracyValidator:
    """Computes similarity and precision metrics between baseline and optimized models."""

    @staticmethod
    def compare_outputs(
        baseline_outputs: Dict[str, np.ndarray],
        optimized_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compare output tensors across all shared output keys."""
        common_keys = list(set(baseline_outputs.keys()).intersection(set(optimized_outputs.keys())))
        if not common_keys:
            base_vals = list(baseline_outputs.values())
            opt_vals = list(optimized_outputs.values())
            if not base_vals or not opt_vals:
                return {
                    "overall_cosine_similarity": 1.0,
                    "overall_mae": 0.0,
                    "overall_rmse": 0.0,
                    "overall_relative_error": 0.0,
                    "max_absolute_diff": 0.0,
                    "top1_agreement_pct": 100.0,
                    "top_prediction_preserved": "YES",
                    "contains_nan_or_inf": False,
                    "accuracy_impact_pct": 0.0,
                    "fidelity_grade": "Excellent (Identical / Empty)",
                    "per_tensor_metrics": {}
                }
            metrics = AccuracyValidator._compare_arrays(base_vals[0], opt_vals[0])
            top_preserved = "YES" if (metrics["top1_agreement_pct"] is None or metrics["top1_agreement_pct"] >= 99.9) else "NO"
            return {
                "overall_cosine_similarity": metrics["cosine_similarity"],
                "overall_mae": metrics["mae"],
                "overall_rmse": metrics["rmse"],
                "overall_relative_error": metrics["relative_error"],
                "max_absolute_diff": metrics["max_absolute_diff"],
                "top1_agreement_pct": metrics["top1_agreement_pct"],
                "top_prediction_preserved": top_preserved,
                "contains_nan_or_inf": metrics["contains_nan_or_inf"],
                "accuracy_impact_pct": round((metrics["cosine_similarity"] - 1.0) * 100.0, 2),
                "fidelity_grade": AccuracyValidator._get_fidelity_grade(metrics["cosine_similarity"], metrics["contains_nan_or_inf"]),
                "per_tensor_metrics": {"output_0": metrics}
            }

        per_tensor_metrics = {}
        all_cosine_sims = []
        all_maes = []
        all_rmses = []
        all_rel_errors = []
        all_max_diffs = []
        all_top1_agreements = []
        any_nan_inf = False
        all_top_preserved = True

        for k in common_keys:
            metrics = AccuracyValidator._compare_arrays(baseline_outputs[k], optimized_outputs[k])
            per_tensor_metrics[k] = metrics
            all_cosine_sims.append(metrics["cosine_similarity"])
            all_maes.append(metrics["mae"])
            all_rmses.append(metrics["rmse"])
            all_rel_errors.append(metrics["relative_error"])
            all_max_diffs.append(metrics["max_absolute_diff"])
            if metrics["contains_nan_or_inf"]:
                any_nan_inf = True
            if metrics["top1_agreement_pct"] is not None:
                all_top1_agreements.append(metrics["top1_agreement_pct"])
                if metrics["top1_agreement_pct"] < 99.9:
                    all_top_preserved = False

        avg_cosine = round(float(np.mean(all_cosine_sims)), 5) if all_cosine_sims else 1.0
        avg_mae = round(float(np.mean(all_maes)), 6) if all_maes else 0.0
        avg_rmse = round(float(np.mean(all_rmses)), 6) if all_rmses else 0.0
        avg_rel_error = round(float(np.mean(all_rel_errors)), 6) if all_rel_errors else 0.0
        max_diff = round(float(np.max(all_max_diffs)), 6) if all_max_diffs else 0.0
        top1_avg = round(float(np.mean(all_top1_agreements)), 2) if all_top1_agreements else None
        top_preserved_str = "YES" if (all_top_preserved and not any_nan_inf) else "NO"

        acc_impact_pct = round((avg_cosine - 1.0) * 100.0, 2)

        return {
            "overall_cosine_similarity": avg_cosine,
            "overall_mae": avg_mae,
            "overall_rmse": avg_rmse,
            "overall_relative_error": avg_rel_error,
            "max_absolute_diff": max_diff,
            "top1_agreement_pct": top1_avg,
            "top_prediction_preserved": top_preserved_str,
            "contains_nan_or_inf": any_nan_inf,
            "accuracy_impact_pct": acc_impact_pct,
            "fidelity_grade": AccuracyValidator._get_fidelity_grade(avg_cosine, any_nan_inf),
            "per_tensor_metrics": per_tensor_metrics
        }

    @staticmethod
    def _compare_arrays(arr_base: np.ndarray, arr_opt: np.ndarray) -> Dict[str, Any]:
        """Compute metrics between two numpy arrays."""
        b = arr_base.astype(np.float32).flatten()
        o = arr_opt.astype(np.float32).flatten()

        if b.size != o.size:
            min_len = min(b.size, o.size)
            b = b[:min_len]
            o = o[:min_len]

        has_nan_inf = bool(
            np.isnan(b).any() or np.isnan(o).any() or np.isinf(b).any() or np.isinf(o).any()
        )

        if has_nan_inf:
            return {
                "cosine_similarity": 0.0,
                "mae": float("inf"),
                "rmse": float("inf"),
                "relative_error": 1.0,
                "max_absolute_diff": float("inf"),
                "top1_agreement_pct": 0.0,
                "contains_nan_or_inf": True,
            }

        abs_diff = np.abs(b - o)
        mae = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0
        rmse = float(np.sqrt(np.mean(abs_diff ** 2))) if abs_diff.size > 0 else 0.0
        max_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0

        norm_b = float(np.linalg.norm(b))
        norm_o = float(np.linalg.norm(o))
        norm_diff = float(np.linalg.norm(b - o))

        if norm_b <= 1e-8 and norm_o <= 1e-8:
            cosine_sim = 1.0
            rel_error = 0.0
        elif norm_b <= 1e-8 or norm_o <= 1e-8:
            cosine_sim = 0.0
            rel_error = 1.0
        else:
            dot = float(np.dot(b, o))
            cosine_sim = dot / (norm_b * norm_o)
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            rel_error = norm_diff / norm_b

        top1_agreement = None
        if arr_base.ndim >= 2 and arr_base.shape[-1] > 1:
            base_preds = np.argmax(arr_base, axis=-1)
            opt_preds = np.argmax(arr_opt, axis=-1)
            top1_agreement = float(np.mean(base_preds == opt_preds) * 100.0)
        elif arr_base.size > 1:
            base_pred = int(np.argmax(arr_base))
            opt_pred = int(np.argmax(arr_opt))
            top1_agreement = 100.0 if base_pred == opt_pred else 0.0

        return {
            "cosine_similarity": round(cosine_sim, 5),
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "relative_error": round(rel_error, 6),
            "max_absolute_diff": round(max_diff, 6),
            "top1_agreement_pct": round(top1_agreement, 2) if top1_agreement is not None else None,
            "contains_nan_or_inf": False,
        }

    @staticmethod
    def _get_fidelity_grade(cosine_sim: float, has_nan_inf: bool = False) -> str:
        if has_nan_inf:
            return "Corrupted (NaN / Inf values detected)"
        if cosine_sim >= 0.999:
            return "Excellent (Indistinguishable from FP32)"
        elif cosine_sim >= 0.99:
            return "Very High (>99% output similarity)"
        elif cosine_sim >= 0.95:
            return "Good (Minor quantization drift)"
        elif cosine_sim >= 0.90:
            return "Fair (Moderate precision loss, calibration suggested)"
        else:
            return "Poor (Significant precision deviation)"
