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
