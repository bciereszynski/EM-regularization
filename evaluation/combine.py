import pandas as pd

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

julia_results = {}
with open("julia_results.txt", "r") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 7:
            continue
        method, filename, nmi, ari, obj, iters, time_total = parts
        key = (method, filename)
        julia_results[key] = {
            "julia_nmi": float(nmi),
            "julia_ari": float(ari),
            "julia_obj": float(obj),
            "julia_iter": int(iters),
            "julia_time": float(time_total)
        }

all_keys = sorted(set(cpp_results.keys()) | set(julia_results.keys()))
combined = []

for key in all_keys:
    method, filename = key
    cpp = cpp_results.get(key, {})
    julia = julia_results.get(key, {})

    name_core = filename.replace(".csv", "").split("_")
    k = int(name_core[0]) if len(name_core) > 0 else ""
    d = int(name_core[1]) * 10 if len(name_core) > 1 else ""  # 10× multiplier
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
        "julia_time_per_iter": (julia.get("julia_time", 0) / julia.get("julia_iter", 1)) if julia else "",
        "delta_time_per_iter": (
                cpp.get("cpp_time_per_iter", 0) -
                (julia.get("julia_time", 0) / julia.get("julia_iter", 1))
        ) if cpp and julia else "",
        "cpp_obj": cpp.get("cpp_obj", ""),
        "julia_obj": julia.get("julia_obj", ""),
        "delta_obj": (cpp.get("cpp_obj", 0) - julia.get("julia_obj", 0)) if cpp and julia else "",
    }
    combined.append(row)

# Save to Excel
df = pd.DataFrame(combined)
df.to_excel("combined_results.xlsx", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure clean styling
sns.set(style="whitegrid")

# 1. Histogram of delta_time_per_iter
plt.figure(figsize=(8, 5))
sns.histplot(df["delta_time_per_iter"].dropna(), bins=30, kde=True)
plt.title("Histogram of Δ Time per Iteration (C++ - Julia)")
plt.xlabel("Δ Time per Iteration")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_delta_time_per_iter_hist.png")
plt.close()

# 2. Scatter: cpp_time_per_iter vs julia_time_per_iter
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

# 3. Scatter: cpp_nmi vs julia_nmi
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

# 4. Bar: average delta_time_per_iter by method
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

# 5. Box plot: delta_ari per method
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="method", y="delta_ari")
plt.title("Δ ARI (C++ - Julia) by Method")
plt.xlabel("Method")
plt.ylabel("Δ ARI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_delta_ari_boxplot.png")
plt.close()
