"""
notebooks/eda.py
────────────────
EDA — NYC Yellow Taxi 2024-2026
Loads pre-computed stats from cache → runs in <5 seconds.

Build cache first (one time only):
    python scripts/build_cache.py
Then run:
    python notebooks/eda.py
"""

import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})
COLORS   = sns.color_palette("muted")
PLOT_DIR = Path(__file__).parent
CACHE    = Path(__file__).parent.parent / "data" / "cache" / "eda_stats.npz"

def save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 {path.name}")

print("=" * 55)
print("  NYC Yellow Taxi 2024–2026 — EDA")
print("=" * 55)

# ── Load cache ─────────────────────────────────────────────────────────────────
if not CACHE.exists():
    print("\n❌ Cache not found.")
    print("   Run this first:  python scripts/build_cache.py")
    sys.exit(1)

t0   = time.time()
data = np.load(CACHE, allow_pickle=True)

hour_avg    = data["hour_avg"]        # shape (24,)
wday_avg    = data["wday_avg"]        # shape (2, 24)
season_avg  = data["season_avg"]      # shape (12, 24)
yoy_avg     = data["yoy_avg"]         # shape (n_years, 24)
years       = data["years"].tolist()
top15_zones = data["top15_zones"].tolist()
top15_vals  = data["top15_vals"].tolist()
month_keys  = data["month_keys"].tolist()
month_vals  = data["month_vals"].tolist()

total_trips = int(sum(month_vals))
print(f"\n⚡ Cache loaded in {time.time()-t0:.2f}s")
print(f"   Months: {len(month_keys)} | Total trips: {total_trips:,}")


# ═══════════════════════════════════════════════════════
#  7 Plots
# ═══════════════════════════════════════════════════════
print("\nGenerating plots...")

# ── 1. Monthly trip volume ─────────────────────────────
labels = [m[5:] + "\n" + m[:4] for m in month_keys]
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(range(len(month_keys)), month_vals, color=COLORS[0], alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(month_keys)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Monthly NYC Yellow Taxi Trip Volume (2024–2026)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
for i, m in enumerate(month_keys):
    if m.endswith("-01"):
        ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(i + 0.1, max(month_vals) * 0.97, m[:4], fontsize=9, color="gray", fontweight="bold")
plt.tight_layout()
save(fig, "01_monthly_volume.png")

# ── 2. Hourly demand profile ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(24), hour_avg, color=COLORS[1], alpha=0.85, edgecolor="white")
ax.axvspan(7, 9.5, alpha=0.12, color="tomato", label="AM Rush")
ax.axvspan(16, 20, alpha=0.12, color="orange", label="PM Rush")
ax.set_title("Avg Pickup Demand by Hour of Day (2024–2026)", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Trips per Zone-Hour")
ax.set_xticks(range(24))
ax.legend()
plt.tight_layout()
save(fig, "02_hourly_profile.png")

# ── 3. Weekday vs Weekend ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(24), wday_avg[0], label="Weekday", color=COLORS[0], linewidth=2.5, marker="o", markersize=4)
ax.plot(range(24), wday_avg[1], label="Weekend", color=COLORS[2], linewidth=2.5, marker="o", markersize=4)
ax.axvspan(7, 9.5, alpha=0.1, color="tomato", label="AM Rush")
ax.axvspan(16, 20, alpha=0.1, color="orange", label="PM Rush")
ax.set_title("Weekday vs Weekend — Hourly Demand (2024–2026)", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Trips per Zone-Hour")
ax.set_xticks(range(24))
ax.legend()
plt.tight_layout()
save(fig, "03_weekday_vs_weekend.png")

# ── 4. Top 15 zones ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh([str(z) for z in top15_zones], top15_vals,
        color=sns.color_palette("Blues_r", 15))
ax.set_title("Top 15 Pickup Zones — Total Demand (2024–2026)", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Pickups")
ax.set_ylabel("Zone ID")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
plt.tight_layout()
save(fig, "04_top_zones.png")

# ── 5. Demand by hour — log scale ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(24), hour_avg, color=COLORS[3], edgecolor="white")
axes[0].set_title("Avg Demand by Hour (linear)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Avg Trips per Zone-Hour")
axes[0].set_xticks(range(24))

axes[1].bar(range(24), hour_avg, color=COLORS[4], edgecolor="white", log=True)
axes[1].set_title("Avg Demand by Hour (log scale)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Avg Trips (log)")
axes[1].set_xticks(range(24))
plt.tight_layout()
save(fig, "05_demand_linear_vs_log.png")

# ── 6. Seasonal heatmap ────────────────────────────────
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
# Only show months that have data (non-zero rows)
has_data = season_avg.sum(axis=1) > 0
season_df = pd.DataFrame(
    season_avg[has_data],
    index=[month_labels[i] for i in range(12) if has_data[i]],
    columns=range(24),
)
fig, ax = plt.subplots(figsize=(18, 5))
sns.heatmap(season_df, ax=ax, cmap="YlOrRd", linewidths=0.3,
            cbar_kws={"label": "Avg Trips per Zone-Hour"},
            fmt=".0f", annot=True, annot_kws={"size": 7})
ax.set_title("Seasonal Heatmap — Avg Demand by Month × Hour (2024–2026)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Month")
plt.tight_layout()
save(fig, "06_seasonal_heatmap.png")

# ── 7. Year-over-year comparison ───────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
palette = sns.color_palette("tab10", len(years))
for i, (yr, row) in enumerate(zip(years, yoy_avg)):
    ax.plot(range(24), row, label=str(yr), color=palette[i],
            linewidth=2.5, marker="o", markersize=3)
ax.set_title("Year-over-Year Hourly Demand (2024 vs 2025 vs 2026)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Trips per Zone-Hour")
ax.set_xticks(range(24))
ax.legend(title="Year")
plt.tight_layout()
save(fig, "07_year_over_year.png")

# ── Summary ────────────────────────────────────────────
peak_idx = int(np.argmax(month_vals))
slow_idx = int(np.argmin(month_vals))
print(f"\n{'='*55}")
print(f"  ✅ Done in {time.time()-t0:.1f}s — 7 plots saved to notebooks/")
print(f"{'='*55}")
print(f"  Total trips:   {total_trips:,}")
print(f"  Peak month:    {month_keys[peak_idx]}  ({month_vals[peak_idx]:,})")
print(f"  Slowest month: {month_keys[slow_idx]}  ({month_vals[slow_idx]:,})")
print(f"  Peak hour:     {int(np.argmax(hour_avg))}:00")
print(f"\n  ▶ Next: python src/train.py")
