import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from jsp_rl.utils import generate_jsp_instance
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_20x20.yaml")
    parser.add_argument("--out", type=str, default="data/val_dataset_20x20.pt")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def make_eval_env(instance, left_shift=True):
    return DisjunctiveGraphJspEnv(
        jps_instance=instance,
        perform_left_shift_if_possible=left_shift,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
        reward_function="zero",
    )


def remaining_work(instance, op_id):
    durations = instance[1]
    n_machines = instance.shape[2]
    job_id = op_id // n_machines
    op_pos = op_id % n_machines
    return int(durations[job_id, op_pos:].sum())


def remaining_ops(instance, op_id):
    n_machines = instance.shape[2]
    op_pos = op_id % n_machines
    return n_machines - op_pos


def processing_time(instance, op_id):
    durations = instance[1]
    n_machines = instance.shape[2]
    return int(durations[op_id // n_machines, op_id % n_machines])


def select_action(instance, valid_actions, rule):
    if rule == "spt":
        return min(valid_actions, key=lambda op: processing_time(instance, op))

    if rule == "mwkr":
        return max(valid_actions, key=lambda op: remaining_work(instance, op))

    if rule == "fdd_mwkr":
        return min(
            valid_actions,
            key=lambda op: processing_time(instance, op)
            / max(remaining_work(instance, op), 1),
        )

    if rule == "mopnr":
        return max(valid_actions, key=lambda op: remaining_ops(instance, op))

    raise ValueError(f"Unknown heuristic rule: {rule}")


def rollout_dispatching_rule(instance, rule, left_shift=True):
    env = make_eval_env(instance, left_shift=left_shift)
    obs, _ = env.reset()

    done = False
    truncated = False

    while not (done or truncated):
        mask = env.valid_action_mask()
        valid_actions = np.flatnonzero(mask)

        action = int(select_action(instance, valid_actions, rule))
        obs, reward, done, truncated, info = env.step(action)

    return {"makespan": int(info["makespan"])}


def rollout_random_policy(instance, seed=42, left_shift=True):
    rng = np.random.default_rng(seed)

    env = make_eval_env(instance, left_shift=left_shift)
    obs, _ = env.reset()

    done = False
    truncated = False

    while not (done or truncated):
        mask = env.valid_action_mask()
        valid_actions = np.flatnonzero(mask)

        action = int(rng.choice(valid_actions))
        obs, reward, done, truncated, info = env.step(action)

    return {"makespan": int(info["makespan"])}


def build_validation_instances(cfg, seed):
    rng = random.Random(seed + 999)

    return [
        generate_jsp_instance(
            n_jobs=cfg["data"]["n_jobs"],
            n_machines=cfg["data"]["n_machines"],
            min_duration=cfg["data"]["min_duration"],
            max_duration=cfg["data"]["max_duration"],
            rng=rng,
        )
        for _ in range(cfg["data"]["n_val_instances"])
    ]


def benchmark_instances(instances, rules, seed):
    per_instance = []

    for i, instance in enumerate(instances):
        print(f"Benchmarking instance {i + 1}/{len(instances)}")

        item = {
            "instance_id": i,
            "random": rollout_random_policy(
                instance,
                seed=seed + i,
                left_shift=True,
            )["makespan"],
        }

        for rule in rules:
            item[rule] = rollout_dispatching_rule(
                instance,
                rule=rule,
                left_shift=True,
            )["makespan"]

        per_instance.append(item)

    return per_instance


def build_summary(per_instance, rules):
    summary = {
        "random": summarize([x["random"] for x in per_instance])
    }

    for rule in rules:
        summary[rule] = summarize([x[rule] for x in per_instance])

    return summary


def save_dataset(cfg, instances, per_instance, summary, out_path):
    data = {
        "config": cfg,
        "instances": instances,
        "per_instance": per_instance,
        "summary": summary,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, out_path)

    return out_path

if __name__ == "__main__":

    # Load configuration and validation settings.
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])

    # Generate a fixed validation dataset.
    instances = build_validation_instances(cfg, seed)

    # Evaluate all baseline dispatching rules.
    rules = ["spt", "mwkr", "fdd_mwkr", "mopnr"]
    per_instance = benchmark_instances(
        instances,
        rules,
        seed,
    )

    # Compute summary statistics.
    summary = build_summary(
        per_instance,
        rules,
    )

    # Save dataset, per-instance results, and summary.
    out_path = save_dataset(
        cfg=cfg,
        instances=instances,
        per_instance=per_instance,
        summary=summary,
        out_path=args.out,
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")