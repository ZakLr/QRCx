def print_summary(results: dict) -> None:
    """Pretty-print summary table to terminal.

    Args:
        results: Nested dict of {model: {horizon: {metric: value}}}.
    """
    header = f"{'Model':<20} {'RMSE 1h':<10} {'MAE 1h':<10} {'Skill 1h':<10} {'RMSE 6h':<10} {'MAE 6h':<10} {'Skill 6h':<10} {'FSDH':<8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for model, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        rmse_1 = _get(metrics, 1, "rmse")
        mae_1 = _get(metrics, 1, "mae")
        skill_1 = _get(metrics, 1, "skill")
        rmse_6 = _get(metrics, 6, "rmse")
        mae_6 = _get(metrics, 6, "mae")
        skill_6 = _get(metrics, 6, "skill")
        fsdh_val = _get(metrics, 1, "fsdh", default="")

        print(
            f"{model:<20} {rmse_1:<10} {mae_1:<10} {skill_1:<10} "
            f"{rmse_6:<10} {mae_6:<10} {skill_6:<10} {fsdh_val:<8}"
        )

    print(sep + "\n")


def _get(metrics: dict, horizon: int, key: str, default: str = "N/A") -> str:
    h = metrics.get(horizon, {})
    val = h.get(key, default)
    if isinstance(val, (int, float)):
        return f"{val:.4f}"
    return str(val)
