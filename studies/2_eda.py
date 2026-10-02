from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader.featured import load_featured_data

# Set visual styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Load data
data_path = load_featured_data()
df = pd.read_csv(data_path)

# 2. Target & Continuous Distributions (4-panel overview)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target Balance
sns.countplot(data=df, x='Churn', palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Churn (Target Variable)", fontsize=13, fontweight='bold')
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{p.get_height()} ({p.get_height()/len(df)*100:.1f}%)',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')

# Tenure by Churn
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=30, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title("Tenure Distribution by Churn", fontsize=13, fontweight='bold')

# MonthlyCharges Density
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title("Monthly Charges Density by Churn", fontsize=13, fontweight='bold')

# Active Services vs Churn
service_churn = pd.crosstab(df['active_services'], df['Churn'], normalize='index') * 100
service_churn.plot(kind='bar', stacked=True, color=['#66c2a5', '#fc8d62'], ax=axes[1, 1])
axes[1, 1].set_title("Churn Rate by Number of Active Services (%)", fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel("Percentage")
axes[1, 1].set_xlabel("Number of Active Services")
axes[1, 1].legend(title="Churn", loc="upper right")

plt.tight_layout()
plt.show()

# 3. Categorical Risk Drivers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, col in enumerate(['Contract', 'InternetService', 'PaymentMethod']):
    ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, color=['#66c2a5', '#fc8d62'], ax=axes[idx])
    axes[idx].set_title(f"Churn by {col} (%)", fontsize=12, fontweight='bold')
    axes[idx].set_ylabel("Percentage")
    axes[idx].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()

# 4. Correlation Heatmap
df_corr = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']].copy()
df_corr['Churn_numeric'] = (df['Churn'] == 'Yes').astype(int)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()