# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)

# %%
# 1) OpenAI / Anthropic GPU park estimate

companies = {
    "OpenAI": {
        "total_low": 900_000,
        "total_base": 1_100_000,
        "total_high": 1_400_000,
        "training_share_low": 0.40,
        "training_share_base": 0.48,
        "training_share_high": 0.55,
    },
    "Anthropic": {
        "total_low": 250_000,
        "total_base": 350_000,
        "total_high": 450_000,
        "training_share_low": 0.25,
        "training_share_base": 0.32,
        "training_share_high": 0.40,
    },
}

rows = []
for company, a in companies.items():
    for scenario in ["low", "base", "high"]:
        total = a[f"total_{scenario}"]
        train_share = a[f"training_share_{scenario}"]
        rows.append({
            "company": company,
            "scenario": scenario,
            "total_gpu_eq": total,
            "training_gpu_eq": total * train_share,
            "inference_gpu_eq": total * (1 - train_share),
            "training_share": train_share,
            "inference_share": 1 - train_share,
        })

gpu_df = pd.DataFrame(rows)
gpu_df

# %%
def fmt_m(x):
    return f"{x/1e6:.2f}M"

summary = []
for company, a in companies.items():
    summary.append({
        "company": company,
        "total_low": fmt_m(a["total_low"]),
        "total_base": fmt_m(a["total_base"]),
        "total_high": fmt_m(a["total_high"]),
        "training_base": fmt_m(a["total_base"] * a["training_share_base"]),
        "inference_base": fmt_m(a["total_base"] * (1 - a["training_share_base"])),
    })

pd.DataFrame(summary)

# %%
pivot = gpu_df.pivot(index="company", columns="scenario", values=["total_gpu_eq", "training_gpu_eq", "inference_gpu_eq"])
pivot

# %%
# 2) Atlas Cloud margin analysis

price_input = 1.68
price_output = 3.38

# Cost-to-serve assumptions per 1M tokens
cost_input_low, cost_input_base, cost_input_high = 0.30, 0.50, 0.80
cost_output_low, cost_output_base, cost_output_high = 0.90, 1.50, 2.20

margin_table = pd.DataFrame([
    ["Input", price_input, cost_input_low, cost_input_base, cost_input_high],
    ["Output", price_output, cost_output_low, cost_output_base, cost_output_high],
], columns=["Metric", "Price", "Cost Low", "Cost Base", "Cost High"])

margin_table["Gross Margin Low"] = (margin_table["Price"] - margin_table["Cost High"]) / margin_table["Price"]
margin_table["Gross Margin Base"] = (margin_table["Price"] - margin_table["Cost Base"]) / margin_table["Price"]
margin_table["Gross Margin High"] = (margin_table["Price"] - margin_table["Cost Low"]) / margin_table["Price"]

margin_table

# %%
def blended_margin(input_share, output_share, cost_input, cost_output):
    revenue = input_share * price_input + output_share * price_output
    cost = input_share * cost_input + output_share * cost_output
    return (revenue - cost) / revenue

mixes = [
    {"name": "80/20 input/output", "input_share": 0.8, "output_share": 0.2},
    {"name": "50/50 input/output", "input_share": 0.5, "output_share": 0.5},
    {"name": "20/80 input/output", "input_share": 0.2, "output_share": 0.8},
]

rows = []
for mix in mixes:
    for scenario, ci, co in [
        ("low cost", cost_input_low, cost_output_low),
        ("base cost", cost_input_base, cost_output_base),
        ("high cost", cost_input_high, cost_output_high),
    ]:
        gm = blended_margin(mix["input_share"], mix["output_share"], ci, co)
        rows.append({
            "mix": mix["name"],
            "scenario": scenario,
            "gross_margin": gm
        })

sensitivity_df = pd.DataFrame(rows)
sensitivity_df.pivot(index="mix", columns="scenario", values="gross_margin").round(3)

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
plot_df = sensitivity_df.copy()
plot_df["gross_margin_pct"] = plot_df["gross_margin"] * 100

for scenario in ["low cost", "base cost", "high cost"]:
    subset = plot_df[plot_df["scenario"] == scenario]
    ax.plot(subset["mix"], subset["gross_margin_pct"], marker="o", linewidth=2, label=scenario)

ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("Gross margin, %")
ax.set_xlabel("Token mix")
ax.set_title("Atlas Cloud margin sensitivity on DeepSeek V4 Pro")
ax.legend()
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
base_mix = {"input_share": 0.7, "output_share": 0.3}
base_gm = blended_margin(base_mix["input_share"], base_mix["output_share"], cost_input_base, cost_output_base)

result = {
    "base_mix": base_mix,
    "base_gross_margin": round(base_gm, 4),
    "interpretation": "profitable" if base_gm > 0 else "breakeven_or_loss"
}
result
