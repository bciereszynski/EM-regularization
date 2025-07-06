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
    with open("results.txt", "r") as f:
        lines = f.readlines()

    with open("summary_results.txt", "w") as out:
        # Optional: write header
        out.write("METHOD FILE OBJECTIVE ARI NMI ITERATIONS TIME TIME_PER_ITER\n")

        for line in lines:
            try:
                data = parse_line(line)
                ari = adjusted_rand_score(data["expected"], data["assigned"])
                nmi = normalized_mutual_info_score(data["expected"], data["assigned"])
                time_per_iter = data["time_total"] / data["iterations"] if data["iterations"] else 0.0

                out.write(f"{data['method']} {data['filename']} {data['objective']:.6f} "
                          f"{ari:.6f} {nmi:.6f} {data['iterations']} "
                          f"{data['time_total']:.6f} {time_per_iter:.6f}\n")
            except Exception as e:
                print(f"Error processing line:\n{line.strip()}\nError: {e}")


if __name__ == "__main__":
    main()
