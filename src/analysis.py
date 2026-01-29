from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


"""
расчет продуктовых метрик, статистические проверки и формирование выводов

продукт метрики
1) преобразуем, чтоб была табличка: пользователь, группа, заплатил ли, что купил
2) база кпи: конверсия, средний доход на пользователя и платящего пользователя
3) воронка: пользователи дошедшие до текущего/пользователи на предыдущем; пользователи на текущем/на первом.
3) удержание: когорта -- кол-во пользователей в один день
ретеншн и средний взвешенный ртеншин, колво зашедших в день N/размер когорты в день 0; обхединение когорт в среднее

статистика:
1) Хи-квадрат 
2) тест на конверсию Z-тест, 95% доверит. интервал
3) сравнение выручки (тк доход распределён неравномерно, вычисляем доверит интервал)

"""

# =============================================================================
# Metrics
# =============================================================================
@dataclass(frozen=True)
class FunnelResult:

    steps: List[str]
    by_variant: pd.DataFrame


def _ensure_ts_and_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "ts" not in d.columns:
        raise ValueError("Missing column 'ts'")
    d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
    if d["ts"].isna().any():
        raise ValueError("Column 'ts' has invalid datetime values")
    d["date"] = d["ts"].dt.floor("D")
    return d


def build_user_table(df: pd.DataFrame, pay_event: str = "pay") -> pd.DataFrame:

    d = _ensure_ts_and_date(df)

    if (d["event"] == "signup").any():
        signup_ts = d.loc[d["event"] == "signup"].groupby("user_id")["ts"].min()
    else:
        signup_ts = d.groupby("user_id")["ts"].min()

    pay_mask = d["event"] == pay_event
    revenue = d.loc[pay_mask].groupby("user_id")["amount"].sum() if "amount" in d.columns else pd.Series(dtype=float)
    first_pay_ts = d.loc[pay_mask].groupby("user_id")["ts"].min() if pay_mask.any() else pd.Series(dtype="datetime64[ns]")

    variant = d.groupby("user_id")["variant"].agg(lambda x: str(x.iloc[0]).upper().strip())

    users = pd.DataFrame({"user_id": variant.index, "variant": variant.values})
    users["signup_ts"] = users["user_id"].map(signup_ts)
    users["revenue"] = users["user_id"].map(revenue).fillna(0.0)
    users["is_payer"] = users["revenue"] > 0
    users["first_pay_ts"] = users["user_id"].map(first_pay_ts)

    return users


def compute_basic_kpis(users: pd.DataFrame) -> pd.DataFrame:
    g = users.groupby("variant").agg(
        n_users=("user_id", "nunique"),
        paying_users=("is_payer", "sum"),
        total_revenue=("revenue", "sum"),
        arpu=("revenue", "mean"),
    )
    g["conversion"] = g["paying_users"] / g["n_users"]
    g["arppu"] = (g["total_revenue"] / g["paying_users"].replace(0, np.nan)).fillna(0.0)
    return g.reset_index()


def compute_funnel(df: pd.DataFrame, steps: List[str]) -> FunnelResult:
    d = _ensure_ts_and_date(df)
    variants = ["A", "B"]
    rows = []

    for v in variants:
        dv = d[d["variant"] == v]
        reached = {step: set(dv.loc[dv["event"] == step, "user_id"].unique()) for step in steps}

        cum: List[Tuple[str, int]] = []
        current: Optional[set] = None
        for step in steps:
            current = reached[step] if current is None else current.intersection(reached[step])
            cum.append((step, len(current)))

        base = cum[0][1] if cum else 0
        prev: Optional[int] = None
        for step, cnt in cum:
            step_conv = (cnt / prev) if (prev and prev > 0) else (1.0 if prev is None else 0.0)
            overall = (cnt / base) if base > 0 else 0.0
            rows.append(
                {"variant": v, "step": step, "users_reached": cnt, "step_conv": step_conv, "overall_conv": overall}
            )
            prev = cnt

    out = pd.DataFrame(rows)
    return FunnelResult(steps=steps, by_variant=out)


def compute_retention_curve(
    df: pd.DataFrame,
    active_event: Optional[str] = None,
    max_day: int = 14,
) -> pd.DataFrame:
    d = _ensure_ts_and_date(df)

    if (d["event"] == "signup").any():
        cohort_day = d.loc[d["event"] == "signup"].groupby("user_id")["date"].min()
    else:
        cohort_day = d.groupby("user_id")["date"].min()

    if active_event is None:
        act = d.groupby(["user_id", "date"]).size().reset_index(name="n")
    else:
        act = d.loc[d["event"] == active_event].groupby(["user_id", "date"]).size().reset_index(name="n")

    variant = d.groupby("user_id")["variant"].agg(lambda x: str(x.iloc[0]).upper().strip())
    users = pd.DataFrame({"user_id": variant.index, "variant": variant.values})
    users["cohort_day"] = users["user_id"].map(cohort_day)

    act = act.merge(users, on="user_id", how="left")
    act["day_offset"] = (act["date"] - act["cohort_day"]).dt.days
    act = act[(act["day_offset"] >= 0) & (act["day_offset"] <= max_day)]

    cohort_sizes = users.groupby(["variant", "cohort_day"])["user_id"].nunique().rename("cohort_size").reset_index()
    active_counts = (
        act.groupby(["variant", "cohort_day", "day_offset"])["user_id"]
        .nunique()
        .rename("active_users")
        .reset_index()
    )

    m = active_counts.merge(cohort_sizes, on=["variant", "cohort_day"], how="left")
    m["retention"] = m["active_users"] / m["cohort_size"]
    m["w"] = m["cohort_size"]

    out = (
        m.groupby(["variant", "day_offset"])
        .apply(lambda g: np.average(g["retention"], weights=g["w"]))
        .rename("retention")
        .reset_index()
        .rename(columns={"day_offset": "day"})
    )

    return out


# =============================================================================
# Stats
# =============================================================================
@dataclass(frozen=True)
class SRMResult:
    ok: bool
    p_value: float
    observed: Dict[str, int]


def check_srm(users: pd.DataFrame) -> SRMResult:
    counts = users.groupby("variant")["user_id"].nunique().to_dict()
    a = int(counts.get("A", 0))
    b = int(counts.get("B", 0))

    total = a + b
    if total == 0:
        return SRMResult(ok=False, p_value=0.0, observed={"A": 0, "B": 0})

    observed = np.array([a, b], dtype=float)
    expected = np.array([total / 2.0, total / 2.0], dtype=float)

    _, p = stats.chisquare(f_obs=observed, f_exp=expected)
    return SRMResult(ok=(p > 0.01), p_value=float(p), observed={"A": a, "B": b})


@dataclass(frozen=True)
class ConversionTestResult:
    p_value: float
    conv_a: float
    conv_b: float
    abs_diff: float
    rel_lift: float
    ci95_abs: Tuple[float, float]


def conversion_ztest_with_ci(users: pd.DataFrame) -> ConversionTestResult:
    res = users.groupby("variant")["is_payer"].agg(["sum", "count"])
    
    for v in ["A", "B"]:
        if v not in res.index:
            res.loc[v] = [0, 0]
            
    nA, cA = int(res.loc["A", "count"]), int(res.loc["A", "sum"])
    nB, cB = int(res.loc["B", "count"]), int(res.loc["B", "sum"])

    p1, p2 = cA / nA if nA else 0, cB / nB if nB else 0
    diff = p2 - p1
    
    p_pool = (cA + cB) / (nA + nB) if (nA + nB) else 0
    if p_pool == 0 or p_pool == 1 or (nA + nB) == 0:
        pval = 1.0
    else:
        se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1/nA + 1/nB))
        z_score = diff / se_pooled
        pval = 2 * (1 - stats.norm.cdf(abs(z_score)))

    se_diff = np.sqrt(p1*(1-p1)/nA + p2*(1-p2)/nB) if (nA > 0 and nB > 0) else 0
    ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    return ConversionTestResult(
        p_value=float(pval),
        conv_a=float(p1),
        conv_b=float(p2),
        abs_diff=float(diff),
        rel_lift=float(p2/p1 - 1) if p1 > 0 else 0,
        ci95_abs=(float(ci[0]), float(ci[1])),
    )

@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    diff_mean: float
    ci95: Tuple[float, float]
    p_value_two_sided: float
    n_boot: int


def _bootstrap_diff_means(xA: np.ndarray, xB: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    nA, nB = len(xA), len(xB)

    for i in range(n_boot):
        sA = rng.choice(xA, size=nA, replace=True)
        sB = rng.choice(xB, size=nB, replace=True)
        diffs[i] = sB.mean() - sA.mean()

    p = 2.0 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    p = float(min(1.0, p))
    return diffs, p


def arpu_bootstrap(users: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> BootstrapResult:
    xA = users.loc[users["variant"] == "A", "revenue"].to_numpy(dtype=float)
    xB = users.loc[users["variant"] == "B", "revenue"].to_numpy(dtype=float)

    if len(xA) == 0 or len(xB) == 0:
        return BootstrapResult(metric="ARPU", diff_mean=0.0, ci95=(0.0, 0.0), p_value_two_sided=1.0, n_boot=int(n_boot))

    diffs, p = _bootstrap_diff_means(xA, xB, n_boot=n_boot, seed=seed)
    lo, hi = np.quantile(diffs, [0.025, 0.975])

    return BootstrapResult(
        metric="ARPU",
        diff_mean=float(diffs.mean()),
        ci95=(float(lo), float(hi)),
        p_value_two_sided=float(p),
        n_boot=int(n_boot),
    )


# =============================================================================
# Verdict
# =============================================================================
@dataclass(frozen=True)
class Verdict:
    title: str
    body: str


def generate_verdict(
    kpis: pd.DataFrame,
    srm: SRMResult,
    conv_test: ConversionTestResult,
    arpu_boot: Optional[BootstrapResult] = None,
    alpha: float = 0.05,
) -> Verdict:
    k = kpis.set_index("variant")
    conv_a = float(k.loc["A", "conversion"])
    conv_b = float(k.loc["B", "conversion"])
    arpu_a = float(k.loc["A", "arpu"])
    arpu_b = float(k.loc["B", "arpu"])

    lines: List[str] = []

    if not srm.ok:
        lines.append(
            f"⚠️ SRM: сплит трафика подозрительный (p={srm.p_value:.4g}, A={srm.observed.get('A')}, B={srm.observed.get('B')}). "
            "Статвыводы могут быть некорректны до выяснения причин."
        )

    conv_sig = conv_test.p_value < alpha
    lines.append(
        f"Конверсия в оплату: A={conv_a:.2%}, B={conv_b:.2%}. "
        f"Разница (B−A)={conv_test.abs_diff:+.2%} (95% CI {conv_test.ci95_abs[0]:+.2%}..{conv_test.ci95_abs[1]:+.2%}), "
        f"p={conv_test.p_value:.4g}."
    )

    lines.append(f"ARPU: A={arpu_a:.2f}, B={arpu_b:.2f} (разница {arpu_b - arpu_a:+.2f}).")

    arpu_sig = None
    if arpu_boot is not None:
        arpu_sig = (arpu_boot.ci95[0] > 0) or (arpu_boot.ci95[1] < 0)
        lines.append(
            f"Bootstrap ARPU diff (B−A): mean={arpu_boot.diff_mean:+.2f}, 95% CI {arpu_boot.ci95[0]:+.2f}..{arpu_boot.ci95[1]:+.2f}, "
            f"p≈{arpu_boot.p_value_two_sided:.4g} (n={arpu_boot.n_boot})."
        )

    if not srm.ok:
        recommendation = "Сначала починить/объяснить SRM, затем повторить эксперимент."
    else:
        if arpu_boot is not None and arpu_sig:
            recommendation = (
                "Рекомендуется выкатывать B (ARPU статистически выше)."
                if arpu_b > arpu_a
                else "Рекомендуется НЕ выкатывать B (ARPU статистически ниже)."
            )
        else:
            if conv_sig and (conv_b > conv_a):
                recommendation = "Рекомендуется выкатывать B (конверсия статистически выше)."
            elif conv_sig and (conv_b < conv_a):
                recommendation = "Рекомендуется НЕ выкатывать B (конверсия статистически ниже)."
            else:
                recommendation = "Разница неубедительна: собрать больше данных или сменить гипотезу/метрику."

    lines.append("")
    lines.append(f"**Рекомендация:** {recommendation}")

    return Verdict(title="Auto-Report", body="\n\n".join(lines))
