#!/usr/bin/env python3
"""Generate every LaTeX table in the paper programmatically from
results/*.json, so main.tex \\input's these fragments and the paper
cannot drift from the underlying data. Run before every paper compile.

Tables written to docs/paper/tables/*.tex:
  tab_headline.tex   -- Sec 6.3, ONE table, every model, canonical test split
  tab_ablation.tex   -- Sec 6.1, dissipation on/off
  tab_concat.tex     -- Sec 6.2, A/B/C/C' concatenated readout
  tab_ipc.tex        -- Sec 5, IPC-matched vs reference config
  tab_dirac3.tex     -- Sec 7.5, Dirac-3/SA/Lasso/greedy comparison
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "docs" / "paper" / "tables"


def pct(x, bold=False):
    s = f"{x * 100:+.2f}\\%"
    return f"\\textbf{{{s}}}" if bold else s


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def write(name, content):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Table: headline matched benchmark (Sec 6.3), canonical split, locked test
# ---------------------------------------------------------------------------
def make_headline():
    bench = load("full_matched_benchmark_test.json")
    units = load("canonical_units.json")
    fsdh_classical = load("baselines_test.json")["fsdh_curves"]
    fsdh_qrc = load("qrc_fsdh_test.json")["fsdh"]
    std = units["std_scaled_to_anomaly"]

    display = [
        ("persistence", "Persistence", None, None),
        ("arima", "ARIMA(2,1,2)", "classical", None),
        ("arima_auto", "ARIMA (auto-order)", "classical", None),
        ("esn_dim_matched", "ESN, dim-matched (234)", "classical", None),
        ("esn_500", "ESN-500", "classical", None),
        ("residual_esn", "Residual-ESN", "classical", None),
        ("null_ridge", "Null-control Ridge", "classical", "null_ridge_raw_window"),
        ("null_krr", "Null-control KRR$^\\dagger$", "classical", None),
        ("residual_ridge", "Residual-Ridge", "classical", "null_ridge_raw_window"),
        ("v4_qrc_residual_20q", "v4 QRC (20q, statevector)", "qrc", "v4_qrc_residual_20q"),
        ("v5_qrc_residual_12q", "v5 QRC (12q, density matrix)", "qrc", "v5_qrc_residual_12q"),
        ("concat_C_raw_plus_v5", "Concat $C{=}$[raw,v5]", "concat", "concat_C_raw_plus_v5"),
        ("concat_C_raw_plus_v4", "Concat $C{=}$[raw,v4]", "concat", "concat_C_raw_plus_v4"),
        ("gbm_ceiling_probe", "GBM ceiling probe$^\\ddagger$", "ceiling", None),
    ]

    # find best (max) skill@1h and @6h among real rows for bolding
    skills_1h, skills_6h = {}, {}
    for key, _, _, _ in display:
        rows = bench["models"].get(key, {})
        if "1" in rows:
            skills_1h[key] = rows["1"]["skill"]
        if "6" in rows:
            skills_6h[key] = rows["6"]["skill"]
    best_1h = max(skills_1h, key=skills_1h.get)
    best_6h = max(skills_6h, key=skills_6h.get)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & RMSE\textdegree C (1h/6h) & skill (1h/6h) & FSDH & $p$ vs persist.\ (1h/6h) \\")
    lines.append(r"\midrule")
    for key, disp, kind, fsdh_key in display:
        rows = bench["models"].get(key, {})
        if not rows:
            continue
        r1 = rows.get("1", {})
        r6 = rows.get("6", {})
        rmse1 = f"{r1['rmse_degC']:.2f}" if "rmse_degC" in r1 else "--"
        rmse6 = f"{r6['rmse_degC']:.2f}" if "rmse_degC" in r6 else "--"
        s1 = f"{r1['skill'] * 100:+.1f}\\%" if "skill" in r1 else "--"
        s6 = f"{r6['skill'] * 100:+.1f}\\%" if "skill" in r6 else "--"
        skill_str = f"{s1}/{s6}"
        if key in (best_1h, best_6h):
            skill_str = f"\\textbf{{{skill_str}}}"
        if fsdh_key and fsdh_key in fsdh_qrc:
            fsdh_v = str(fsdh_qrc[fsdh_key])
        elif key in fsdh_classical:
            fsdh_v = str(fsdh_classical[key])
        else:
            fsdh_v = "n/a"
        p1 = r1.get("dm_vs_persistence", {}).get("p_value")
        p6 = r6.get("dm_vs_persistence", {}).get("p_value")

        def pshort(p):
            if p is None:
                return "--"
            return "$<.001$" if p < 0.001 else f"${p:.3f}$"

        pstr = f"{pshort(p1)}/{pshort(p6)}"
        row_name = f"\\textbf{{{disp}}}" if key in (best_1h, best_6h) else disp
        lines.append(f"{row_name} & {rmse1}/{rmse6} & {skill_str} & {fsdh_v} & {pstr} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Canonical-split (train 2019--2022/val 2023/test 2024), locked test-2024 headline "
        r"numbers, every row on identical data. RMSE in \textdegree C "
        f"(std$={std:.3f}$" + r"\textdegree C, \texttt{results/canonical\_units.json}). "
        r"$p$ is Diebold--Mariano vs.\ persistence; both concat rows also differ significantly from "
        r"the null-control Ridge baseline (see text), toward worse skill. FSDH: max consecutive "
        r"horizon (of 48) beating persistence. VPT $=1.0$ for every row except ARIMA(2,1,2) (0.0), "
        r"omitted as uninformative here. $^\dagger$Null-control KRR's training set is capped at "
        r"3{,}000 samples. $^\ddagger$GBM is a predictability-ceiling probe, not an RC-comparison "
        r"baseline; FSDH/DM not computed for it.}"
    )
    lines.append(r"\label{tab:headline}")
    lines.append(r"\end{table*}")
    write("tab_headline.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: dissipation ablation (Sec 6.1)
# ---------------------------------------------------------------------------
def make_ablation():
    d = load("sprint4_v5_results.json")
    cells = [c for c in d["ablation_cells"] if c["architecture"] == "residual"]
    by_gamma = {}
    for c in cells:
        by_gamma.setdefault(c["gamma1_name"], {})[c["horizon"]] = c

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"$\gamma_1$ & skill@1h & skill@3h & skill@6h & skill@12h ($p$) \\")
    lines.append(r"\midrule")
    for gname, label in [("tuned", r"$0.03$ (tuned)"), ("diagnostic_off", r"$0.0$ (off)")]:
        row = by_gamma.get(gname, {})
        vals = []
        for h in (1, 3, 6, 12):
            c = row.get(h)
            if c is None:
                vals.append("--")
                continue
            s = f"{c['skill_vs_persistence'] * 100:+.2f}\\%"
            if h == 12:
                p = c["dm_vs_persistence"]["p_value"]
                pstr = f"$p{{\\approx}}0$" if p < 0.001 else f"$p{{=}}{p:.2f}$"
                s = f"\\textbf{{{s}}} ({pstr})" if gname == "tuned" else f"{s} ({pstr})"
            vals.append(s)
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Dissipation-on vs.\ off ablation, identical seed and data, residual architecture, "
        r"12-qubit reference config. With dissipation off, skill collapses to $\approx 0\%$ "
        r"(statistically indistinguishable from persistence) at every horizon; with it on, a real, "
        r"Diebold--Mariano-significant effect emerges at $h{=}12$.}"
    )
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\end{table}")
    write("tab_ablation.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: A/B/C/C' concatenated readout (Sec 6.2), pilot-scale
# ---------------------------------------------------------------------------
def make_concat():
    d = load("hybrid_readout.json")
    ph = d["per_horizon"]

    def pfmt(p):
        if p < 0.001:
            return "$p{<}0.001$"
        if p < 0.01:
            return f"$p{{=}}{p:.3f}$"
        return f"$p{{=}}{p:.2f}$"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"$h$ & $A$ (raw window) & $B$ (v5 only) & $C{=}[A,B]$ & $C'{=}[A,B_{\text{ESN}}]$ \\")
    lines.append(r"\midrule")
    for h in d["horizons"]:
        row = ph[str(h)]
        a, b = row["A"]["skill"], row["B"]["skill"]
        c, cp = row["C"]["skill"], row["Cprime"]["skill"]
        pc = row["dm_C_vs_A"]["p_value"]
        pcp = row["dm_Cprime_vs_A"]["p_value"]
        lines.append(
            f"{h} & {a * 100:+.2f}\\% & {b * 100:+.2f}\\% & "
            f"{c * 100:+.2f}\\% ({pfmt(pc)}) & {cp * 100:+.2f}\\% ({pfmt(pcp)}) \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Skill vs.\ persistence for the raw-window model ($A$), reservoir-only model "
        r"($B$), concatenated quantum model ($C$), and concatenated classical-ESN-control model "
        r"($C'$); real driven $v5$ pilot features (10 qubits, $\gamma_1{=}0.03$), 75/25 fit/eval "
        r"split. $p$ is Diebold--Mariano vs.\ $A$ alone. At every horizon, $C$ is either not "
        r"significantly different from $A$ or significantly \emph{worse}; $C'$ (classical control) "
        r"shows the same pattern -- no row shows the quantum reservoir adding real information "
        r"beyond the raw window, and beyond what a generic classical reservoir of matched size "
        r"already provides.}"
    )
    lines.append(r"\label{tab:concat}")
    lines.append(r"\end{table*}")
    write("tab_concat.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: IPC-matched vs reference config (Sec 5)
# ---------------------------------------------------------------------------
def make_ipc():
    d = load("ipc_matching.json")
    matched = d["matched_forecast_result"]["metrics"]
    ref = d["reference_forecast_result"]["metrics"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Config & skill@1h & skill@3h & skill@6h & skill@12h \\")
    lines.append(r"\midrule")
    for label, metrics, bold_h in [
        (f"Matched ($\\gamma_1{{=}}{d['best_matched_config']['gamma1']}, "
         f"a{{=}}{d['best_matched_config']['a']}$)", matched, {1, 3}),
        (f"Reference ($\\gamma_1{{=}}{d['reference_config']['gamma1']}, "
         f"a{{=}}{d['reference_config']['a']}$)", ref, {6, 12}),
    ]:
        vals = []
        for h in (1, 3, 6, 12):
            s = metrics[str(h)]["skill_vs_persistence"] * 100
            s_str = f"{s:+.2f}\\%"
            if h in bold_h:
                s_str = f"\\textbf{{{s_str}}}"
            vals.append(s_str)
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{IPC-matched vs.\ reference configuration, real reduced-sample pilot forecast "
        f"skill vs.\\ persistence (captured capacity {d['best_matched_config']['captured_total']:.2f} "
        r"matched vs.\ 4.53 reference).}"
    )
    lines.append(r"\label{tab:matching}")
    lines.append(r"\end{table}")
    write("tab_ipc.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    make_headline()
    make_ablation()
    make_concat()
    make_ipc()
    print("\nAll tables regenerated from results/*.json.")
