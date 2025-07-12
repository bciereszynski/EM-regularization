import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === Load C++ Results ===
cpp_results = {}
with open("summary_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        method, filename, obj, nmi, ari, iters, time_total, time_per_iter = parts
        key = (method, filename)
        cpp_results[key] = {
            "cpp_obj": float(obj),
            "cpp_nmi": float(nmi),
            "cpp_ari": float(ari),
            "cpp_iter": int(iters),
            "cpp_time": float(time_total),
            "cpp_time_per_iter": float(time_per_iter)
        }

# === Load Julia Results ===
julia_results = {}
with open("summary_julia_results.txt", "r") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split()]
        if len(parts) != 8:
            continue
        method, filename, obj, nmi, ari, iters, time_total, time_per_iter = parts
        key = (method, filename)
        julia_results[key] = {
            "julia_nmi": float(nmi),
            "julia_ari": float(ari),
            "julia_obj": float(obj),
            "julia_iter": int(iters),
            "julia_time": float(time_total),
            "julia_time_per_iter": float(time_per_iter)
        }

# === Combine Results ===
all_keys = sorted(set(cpp_results.keys()) | set(julia_results.keys()))
combined = []

for key in all_keys:
    method, filename = key
    cpp = cpp_results.get(key, {})
    julia = julia_results.get(key, {})

    name_core = filename.replace(".csv", "").split("_")
    k = int(name_core[0]) if len(name_core) > 0 else ""
    d = int(name_core[1]) if len(name_core) > 1 else ""
    c = float(name_core[2]) if len(name_core) > 2 else ""

    row = {
        "method": method,
        "filename": filename,
        "k": k,
        "d": d,
        "c": c,
        "cpp_nmi": cpp.get("cpp_nmi", ""),
        "julia_nmi": julia.get("julia_nmi", ""),
        "delta_nmi": (cpp.get("cpp_nmi", 0) - julia.get("julia_nmi", 0)) if cpp and julia else "",
        "cpp_ari": cpp.get("cpp_ari", ""),
        "julia_ari": julia.get("julia_ari", ""),
        "delta_ari": (cpp.get("cpp_ari", 0) - julia.get("julia_ari", 0)) if cpp and julia else "",
        "cpp_time_per_iter": cpp.get("cpp_time_per_iter", ""),
        "julia_time_per_iter": julia.get("julia_time_per_iter", ""),
        "delta_time_per_iter": (
                cpp.get("cpp_time_per_iter", 0) -
                julia.get("julia_time_per_iter", 0)
        ) if cpp and julia else "",
        "cpp_obj": cpp.get("cpp_obj", ""),
        "julia_obj": julia.get("julia_obj", ""),
        "delta_obj": (cpp.get("cpp_obj", 0) - julia.get("julia_obj", 0)) if cpp and julia else "",
    }
    combined.append(row)

# === Save Combined Results to Excel ===
df = pd.DataFrame(combined)
df.to_excel("combined_results.xlsx", index=False)

# === Set Plot Style ===
sns.set(style="whitegrid")

# === Plot 1: Histogram of Δ Time per Iteration ===
plt.figure(figsize=(8, 5))
sns.histplot(df["delta_time_per_iter"].dropna(), bins=30, kde=True)
plt.title("Histogram of Δ Time per Iteration (C++ - Julia)")
plt.xlabel("Δ Time per Iteration")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_delta_time_per_iter_hist.png")
plt.close()

# === Plot 2: Scatter of C++ vs Julia Time per Iteration ===
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df, x="cpp_time_per_iter", y="julia_time_per_iter", hue="method")
plt.plot([0, df[["cpp_time_per_iter", "julia_time_per_iter"]].max().max()],
         [0, df[["cpp_time_per_iter", "julia_time_per_iter"]].max().max()],
         'r--', label="x = y")
plt.legend()
plt.xlabel("C++ Time per Iteration")
plt.ylabel("Julia Time per Iteration")
plt.title("C++ vs Julia Time per Iteration")
plt.tight_layout()
plt.savefig("plot_time_per_iter_scatter.png")
plt.close()

# === Plot 3: Scatter of C++ vs Julia NMI ===
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df, x="cpp_nmi", y="julia_nmi", hue="method")
plt.plot([0, 1], [0, 1], 'r--', label="x = y")
plt.xlabel("C++ NMI")
plt.ylabel("Julia NMI")
plt.title("C++ vs Julia NMI")
plt.legend()
plt.tight_layout()
plt.savefig("plot_nmi_scatter.png")
plt.close()

# === Plot 4: Bar of Average Δ Time per Iteration by Method ===
plt.figure(figsize=(8, 5))
avg_time_deltas = df.groupby("method")["delta_time_per_iter"].mean().reset_index()
sns.barplot(data=avg_time_deltas, x="method", y="delta_time_per_iter")
plt.title("Average Δ Time per Iteration by Method")
plt.xlabel("Method")
plt.ylabel("Δ Time per Iteration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_avg_delta_time_per_iter_by_method.png")
plt.close()

# === Plot 5: Box Plot of Δ ARI by Method ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="method", y="delta_ari")
plt.title("Δ ARI (C++ - Julia) by Method")
plt.xlabel("Method")
plt.ylabel("Δ ARI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_delta_ari_boxplot.png")
plt.close()

# === Plot 6: Heatmaps of Δ ARI by d and c for Each Method ===
methods = sorted(df["method"].dropna().unique())
d_vals = sorted(df["d"].dropna().unique())
c_vals = sorted(df["c"].dropna().unique())

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True, sharey=True)

for ax, method in zip(axes.flat, methods):
    subdf = df[df["method"] == method]
    heat_data = subdf.pivot_table(index="c", columns="d", values="delta_ari", aggfunc="mean")
    heat_data = heat_data.reindex(index=c_vals, columns=d_vals)

    sns.heatmap(heat_data, ax=ax, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                vmin=-1.0, vmax=1.0, linewidths=0.5, linecolor="gray")

    ax.set_title(method)
    ax.set_xlabel("features (d)")
    ax.set_ylabel("separability (c)")

plt.suptitle("Δ ARI (C++ - Julia) Heatmaps", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plot_delta_ari_heatmaps.png")
plt.close()

# === Detailed Summary Excel with Stats ===
metrics = ["delta_time_per_iter", "delta_ari", "delta_nmi"]

summary_stats = df.groupby("method")[metrics].agg(["mean", "std", "median", "min", "max", "count"])
# Flatten MultiIndex columns
summary_stats.columns = ["_".join(col) for col in summary_stats.columns]
summary_stats = summary_stats.reset_index()

# Save to Excel
summary_stats.to_excel("avg_deltas_by_method_detailed.xlsx", index=False)
