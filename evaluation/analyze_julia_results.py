from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def parse_line(line):
    parts = line.strip().split(" ")
    if len(parts) < 7:
        raise ValueError("Invalid line format")

    method = parts[0]
    filename = parts[1]
    objective = float(parts[2])
    iterations = int(parts[3])
    time_total = float(parts[4])

    expected = list(map(int, parts[5].strip("[],").split(",")))
    assigned = list(map(int, parts[6].strip("[],").split(",")))

    return {
        "method": method,
        "filename": filename,
        "objective": objective,
        "iterations": iterations,
        "time_total": time_total,
        "expected": expected,
        "assigned": assigned
    }


def main():
    with open("julia_results.txt", "r") as f:
        lines = f.readlines()

    grouped = defaultdict(list)

    # Parse and group by (method, filename)
    for line in lines:
        try:
            data = parse_line(line)
            ari = adjusted_rand_score(data["expected"], data["assigned"])
            nmi = normalized_mutual_info_score(data["expected"], data["assigned"])
            time_per_iter = data["time_total"] / data["iterations"] if data["iterations"] else 0.0

            key = (data["method"], data["filename"])
            grouped[key].append({
                "line": line,
                "objective": data["objective"],
                "ari": ari,
                "nmi": nmi,
                "iterations": data["iterations"],
                "time_total": data["time_total"],
                "time_per_iter": time_per_iter
            })
        except Exception as e:
            print(f"Error processing line:\n{line.strip()}\nError: {e}")

    # Write the run with median ARI for each group
    with open("summary_julia_results.txt", "w") as out:
        for (method, filename), runs in grouped.items():
            if not runs:
                continue

            # Sort by ARI to find median run
            runs_sorted = sorted(runs, key=lambda x: x["ari"])
            median_run = runs_sorted[len(runs_sorted) // 2]

            out.write(f"{method} {filename} {median_run['objective']:.6f} "
                      f"{median_run['nmi']:.6f} {median_run['ari']:.6f} {median_run['iterations']} "
                      f"{median_run['time_total']:.6f} {median_run['time_per_iter']:.6f}\n")


if __name__ == "__main__":
    main()
