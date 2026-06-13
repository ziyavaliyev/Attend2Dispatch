import json
from pathlib import Path

import numpy as np
import torch
from ortools.sat.python import cp_model


def solve_ortools(instance, time_limit=30.0):
    machines = instance[0]
    durations = instance[1]

    n_jobs, n_machines = machines.shape
    horizon = int(durations.sum())

    model = cp_model.CpModel()

    all_tasks = {}
    machine_to_intervals = {m: [] for m in range(n_machines)}

    for j in range(n_jobs):
        for k in range(n_machines):
            m = int(machines[j, k])
            p = int(durations[j, k])

            start = model.NewIntVar(0, horizon, f"start_{j}_{k}")
            end = model.NewIntVar(0, horizon, f"end_{j}_{k}")
            interval = model.NewIntervalVar(start, p, end, f"interval_{j}_{k}")

            all_tasks[(j, k)] = (start, end, interval)
            machine_to_intervals[m].append(interval)

    for m in range(n_machines):
        model.AddNoOverlap(machine_to_intervals[m])

    for j in range(n_jobs):
        for k in range(n_machines - 1):
            model.Add(all_tasks[(j, k + 1)][0] >= all_tasks[(j, k)][1])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan,
        [all_tasks[(j, n_machines - 1)][1] for j in range(n_jobs)],
    )

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 1

    status = solver.Solve(model)

    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    optimal = status == cp_model.OPTIMAL

    result = {
        "makespan": int(solver.Value(makespan)) if feasible else None,
        "status": solver.StatusName(status),
        "optimal": bool(optimal),
        "objective_bound": float(solver.BestObjectiveBound()) if feasible else None,
        "wall_time": float(solver.WallTime()),
    }

    if feasible and not optimal:
        print(
            f"[WARNING] OR-Tools feasible but not proven optimal: "
            f"makespan={result['makespan']}, "
            f"bound={result['objective_bound']}"
        )

    return result


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def add_ortools_to_dataset(dataset_path, out_path=None, time_limit=10.0):
    data = torch.load(dataset_path, weights_only=False)

    instances = data["instances"]
    per_instance = data["per_instance"]

    ortools_makespans = []

    for i, instance in enumerate(instances):
        print(f"Solving OR-Tools {i + 1}/{len(instances)}")

        result = solve_ortools(instance, time_limit=time_limit)
        per_instance[i]["ortools"] = result

        if result["makespan"] is not None:
            ortools_makespans.append(result["makespan"])

    if ortools_makespans:
        data["summary"]["ortools"] = summarize(ortools_makespans)
        data["summary"]["ortools"]["n_feasible"] = len(ortools_makespans)
        data["summary"]["ortools"]["n_optimal"] = int(
            sum(x["ortools"]["optimal"] for x in per_instance)
        )

    save_path = Path(out_path) if out_path else Path(dataset_path)
    torch.save(data, save_path)

    print(json.dumps(data["summary"].get("ortools", {}), indent=2))
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--time_limit", type=float, default=240.0)
    args = parser.parse_args()
    add_ortools_to_dataset(
        dataset_path=args.dataset,
        out_path=args.out,
        time_limit=args.time_limit,
    )