import pandas as pd

cpp_results = {}
with open("summary_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        _, filename, obj, nmi, ari, iter_, time, time_per_iter = parts
        cpp_results[filename] = {
            "filename": filename,
            "cpp_obj": float(obj),
            "cpp_nmi": float(nmi),
            "cpp_ari": float(ari),
            "cpp_iter": int(iter_),
            "cpp_time": float(time),
            "cpp_time_per_iter": float(time_per_iter)
        }

julia_results = {}
with open("julia_results.txt", "r") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 6:
            continue
        filename, nmi, ari, obj, iter_, time = parts
        julia_results[filename] = {
            "filename": filename,
            "julia_nmi": float(nmi),
            "julia_ari": float(ari),
            "julia_obj": float(obj),
            "julia_iter": int(iter_),
            "julia_time": float(time)
        }

all_filenames = sorted(set(cpp_results.keys()) | set(julia_results.keys()))
combined = []

for filename in all_filenames:
    cpp = cpp_results.get(filename, {})
    julia = julia_results.get(filename, {})

    cpp_time_per_iter = cpp.get("cpp_time_per_iter", None)
    julia_iter = julia.get("julia_iter", None)
    julia_time = julia.get("julia_time", None)

    # Parse filename to get k, d, c
    base_name = filename.replace(".csv", "")
    parts = base_name.split("_")
    try:
        k = int(parts[0])
        d = int(parts[1])
        c = float(parts[2])
    except:
        k, d, c = "", "", ""

    row = {
        "filename": filename,
        "k": k,
        "d": d,
        "c": c,
        "cpp_obj": cpp.get("cpp_obj", ""),
        "cpp_nmi": cpp.get("cpp_nmi", ""),
        "cpp_ari": cpp.get("cpp_ari", ""),
        "cpp_iter": cpp.get("cpp_iter", ""),
        "cpp_time": cpp.get("cpp_time", ""),
        "cpp_time_per_iter": cpp_time_per_iter,
        "julia_nmi": julia.get("julia_nmi", ""),
        "julia_ari": julia.get("julia_ari", ""),
        "julia_obj": julia.get("julia_obj", ""),
        "julia_iter": julia_iter,
        "julia_time": julia_time
    }

    row["delta_nmi"] = row["cpp_nmi"] - row["julia_nmi"] if row["cpp_nmi"] != "" and row["julia_nmi"] != "" else ""
    row["delta_ari"] = row["cpp_ari"] - row["julia_ari"] if row["cpp_ari"] != "" and row["julia_ari"] != "" else ""
    if cpp_time_per_iter != "" and julia_iter and julia_time:
        row["delta_time_per_iter"] = cpp_time_per_iter - (julia_time / julia_iter)
    else:
        row["delta_time_per_iter"] = ""

    combined.append(row)

df = pd.DataFrame(combined)

for col in ["cpp_obj", "cpp_nmi", "cpp_ari", "cpp_time", "cpp_time_per_iter",
            "julia_nmi", "julia_ari", "julia_obj", "julia_time", "delta_nmi", "delta_ari", "delta_time_per_iter"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(6)

df.to_excel("combined_results.xlsx", index=False)
