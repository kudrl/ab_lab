from __future__ import annotations

import numpy as np
import pandas as pd


def _lognorm_amount(rng: np.random.Generator, mean=3.0, sigma=1.0) -> float:
    return float(rng.lognormal(mean=mean, sigma=sigma))


"""
б конвертит лучше а
б конвертит хуже, но логнорм кривая
парадокс симпсона 
"""

def _date(day: int) -> str:
    return f"2023-01-{day:02d}T12:00:00"


def generate_conversion_lift(
    n_users: int = 20000,
    base_conv: float = 0.10,
    lift_rel: float = 0.15,
    base_open: float = 0.75,
    max_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=n_users)

    rows = []
    for uid, v in enumerate(variants):
        rows.append([uid, v, _date(1), "signup", 0.0])

        # activity days (open_app)
        n_open_days = rng.integers(0, max(1, max_days // 2) + 1)
        open_days = rng.choice(np.arange(1, max_days + 1), size=n_open_days, replace=False) if n_open_days > 0 else []
        for d in open_days:
            if rng.random() < base_open:
                rows.append([uid, v, _date(int(d)), "open_app", 0.0])

        p = base_conv * (1.0 + lift_rel) if v == "B" else base_conv
        if rng.random() < p:
            pay_day = int(rng.integers(2, max_days + 1))
            rows.append([uid, v, _date(pay_day), "pay", _lognorm_amount(rng)])

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount"])


def generate_arpu_tradeoff(
    n_users: int = 30000,
    conv_a: float = 0.12,
    conv_b: float = 0.10,
    amount_mean_a: float = 2.8,
    amount_mean_b: float = 3.25,
    max_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=n_users)

    rows = []
    for uid, v in enumerate(variants):
        rows.append([uid, v, _date(1), "signup", 0.0])

        p = conv_b if v == "B" else conv_a
        if rng.random() < p:
            mean = amount_mean_b if v == "B" else amount_mean_a
            pay_day = int(rng.integers(2, max_days + 1))
            rows.append([uid, v, _date(pay_day), "pay", _lognorm_amount(rng, mean=mean, sigma=1.0)])

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount"])


def generate_simpson_paradox(
    n_users: int = 40000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=n_users)

    countries = []
    for v in variants:
        if v == "A":
            countries.append(rng.choice(["RU", "DE"], p=[0.7, 0.3]))
        else:
            countries.append(rng.choice(["RU", "DE"], p=[0.3, 0.7]))

    base = {"RU": 0.14, "DE": 0.06}
    lift = 0.20

    rows = []
    for uid, (v, c) in enumerate(zip(variants, countries)):
        rows.append([uid, v, _date(1), "signup", 0.0, c])

        p = base[c] * (1.0 + lift) if v == "B" else base[c]
        if rng.random() < p:
            rows.append([uid, v, _date(2), "pay", _lognorm_amount(rng), c])

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount", "country"])
