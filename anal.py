import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# SLIDE 1: GPU Allocation Estimates (May 2026)
# ==========================================
# Data based on industry estimates (SemiAnalysis, Epoch AI)
gpu_data = {
    'Provider': ['OpenAI', 'Anthropic (NVIDIA)', 'Anthropic (Google TPU)'],
    'Inference (%)': [70, 65, 65],
    'Training (%)': [30, 35, 35]
}
df_gpu = pd.DataFrame(gpu_data)

print("--- Slide 1: GPU Allocation ---")
print(df_gpu)

# ==========================================
# SLIDE 2: Atlas Cloud Unit Economics
# ==========================================
# Pricing per 1 Million Tokens
pricing_data = {
    'Provider': ['Atlas Cloud', 'DeepSeek Official API', 'Hardware Cost (Est.)'],
    'Input Price ($)': [1.68, 0.27, 0.05],
    'Output Price ($)': [3.38, 0.55, 0.25] # HW cost for MoE output is ~$0.15-$0.30
}
df_pricing = pd.DataFrame(pricing_data)

# Margin Calculations
atlas_output = 3.38
ds_official_output = 0.55
hw_cost_output = 0.25

markup_vs_api = atlas_output / ds_official_output
margin_vs_api = (atlas_output - ds_official_output) / atlas_output
margin_vs_hw = (atlas_output - hw_cost_output) / atlas_output

print("\n--- Slide 2: Margin Calculations ---")
print(f"Atlas Cloud Output Price: ${atlas_output}")
print(f"DeepSeek Official Output Price: ${ds_official_output}")
print(f"Markup vs Official API: {markup_vs_api:.1f}x")
print(f"Gross Margin vs API: {margin_vs_api:.1%}")
print(f"Gross Margin vs Hardware Cost: {margin_vs_hw:.1%}")

# Visualization for the Notebook
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_pricing['Provider'], df_pricing['Output Price ($)'], color=['#ff4b4b', '#4bffb5', '#b0b0b0'])
ax.set_ylabel('Price per 1M Output Tokens ($)')
ax.set_title('Output Pricing Comparison: DeepSeek V4 Pro')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()