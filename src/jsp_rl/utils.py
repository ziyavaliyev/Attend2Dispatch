import numpy as np
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import os

# Computes a normalized Critical Lower Bound (CLB) for each node.
# CLB is the earliest possible completion time given precedence constraints in A,
# assuming operations start as early as possible and only job-order dependencies.
def clb(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    dur = X[:, -1].astype(np.float32)          # (T,)
    preds = (A > 0)                             # edge mask
    indeg = preds.sum(axis=0).astype(int)       # (T,)

    q = list(np.where(indeg == 0)[0])
    clb = dur.copy()

    for u in q:                                 # q grows while iterating
        for v in np.where(preds[u])[0]:         # u -> v
            clb[v] = max(clb[v], clb[u] + dur[v])
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(q) != A.shape[0]:
        raise ValueError("A has a cycle (CLB needs a DAG).")

    clb = clb / clb.max()
    clb = clb.reshape(-1, 1)
    return clb

def compute_pna_degree_histogram(data_list):
    max_degree = 0

    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)

    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg

def build_graph_node_features(obs, scheduled, n_machines):
    obs = np.asarray(obs, dtype=np.float32)

    T = obs.shape[0]
    A = obs[:, :T]
    base = obs[:, T:T + n_machines + 1]

    machine_oh = base[:, :n_machines]
    clb_data = clb(A, base).astype(np.float32, copy=False)
    is_scheduled = scheduled.astype(np.float32).reshape(-1, 1)

    x = np.concatenate(
        [machine_oh, is_scheduled, clb_data],
        axis=1,
    ).astype(np.float32)

    return A.astype(np.float32), x

def generate_jsp_instance(
    n_jobs: int,
    n_machines: int,
    min_duration: int = 1,
    max_duration: int = 99,
    rng: random.Random | None = None,
) -> np.ndarray:
    """
    Returns JSP instance with shape [2, n_jobs, n_machines].

    instance[0, j, k] = machine id for operation k of job j
    instance[1, j, k] = processing time
    """
    rng = rng or random.Random()

    machines = []
    durations = []

    for _ in range(n_jobs):
        order = list(range(n_machines))
        rng.shuffle(order)
        machines.append(order)

        durs = [rng.randint(min_duration, max_duration) for _ in range(n_machines)]
        durations.append(durs)

    return np.array([machines, durations], dtype=np.int64)


def operation_id(job: int, op_idx: int, n_machines: int) -> int:
    return job * n_machines + op_idx


def decode_operation_id(op_id: int, n_machines: int) -> tuple[int, int]:
    return op_id // n_machines, op_id % n_machines


def build_initial_state(instance: np.ndarray) -> dict:
    _, n_jobs, n_machines = instance.shape
    return {
        "job_next_op": np.zeros(n_jobs, dtype=np.int64),
        "machine_available": np.zeros(n_machines, dtype=np.int64),
        "job_available": np.zeros(n_jobs, dtype=np.int64),
        "scheduled": np.zeros(n_jobs * n_machines, dtype=np.bool_),
        "machine_sequences": [[] for _ in range(n_machines)],
        "time": 0,
    }


def get_ready_ops(instance: np.ndarray, state: dict) -> list[int]:
    _, n_jobs, n_machines = instance.shape
    ready = []

    for j in range(n_jobs):
        op_idx = int(state["job_next_op"][j])
        if op_idx < n_machines:
            ready.append(operation_id(j, op_idx, n_machines))

    return ready


def apply_action(instance: np.ndarray, state: dict, op_id: int) -> dict:
    machine_order = instance[0]
    proc_times = instance[1]
    _, n_jobs, n_machines = instance.shape

    j, op_idx = decode_operation_id(op_id, n_machines)
    assert op_idx == state["job_next_op"][j], "Invalid action: predecessor not scheduled."

    m = int(machine_order[j, op_idx])
    p = int(proc_times[j, op_idx])

    start = max(int(state["job_available"][j]), int(state["machine_available"][m]))
    finish = start + p

    new_state = {
        "job_next_op": state["job_next_op"].copy(),
        "machine_available": state["machine_available"].copy(),
        "job_available": state["job_available"].copy(),
        "scheduled": state["scheduled"].copy(),
        "machine_sequences": [seq.copy() for seq in state["machine_sequences"]],
        "time": finish,
    }

    new_state["job_next_op"][j] += 1
    new_state["job_available"][j] = finish
    new_state["machine_available"][m] = finish
    new_state["scheduled"][op_id] = True
    new_state["machine_sequences"][m].append(op_id)
    
    return new_state


def is_done(instance: np.ndarray, state: dict) -> bool:
    _, _, n_machines = instance.shape
    return bool(np.all(state["job_next_op"] >= n_machines))


def makespan(state: dict) -> int:
    return int(np.max(state["machine_available"]))


def state_to_tokens(instance: np.ndarray, state: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tokens: [T, 16]
      mask:   [T] boolean, True for valid actions
    """
    machine_order = instance[0]
    proc_times = instance[1]
    _, n_jobs, n_machines = instance.shape
    T = n_jobs * n_machines

    tokens = np.zeros((T, 16), dtype=np.float32)
    mask = np.zeros(T, dtype=np.bool_)

    ready_ops = set(get_ready_ops(instance, state))
    max_duration = max(float(proc_times.max()), 1.0)
    current_makespan = max(float(makespan(state)), 1.0)

    for j in range(n_jobs):
        for k in range(n_machines):
            op_id = operation_id(j, k, n_machines)
            m = int(machine_order[j, k])
            p = int(proc_times[j, k])

            scheduled = float(state["scheduled"][op_id])
            ready = float(op_id in ready_ops)

            predecessor_done = 1.0 if k == 0 else float(state["scheduled"][operation_id(j, k - 1, n_machines)])
            successor_exists = float(k < n_machines - 1)

            job_progress = float(state["job_next_op"][j]) / n_machines
            op_position = float(k) / max(n_machines - 1, 1)
            machine_id_norm = float(m) / max(n_machines - 1, 1)
            job_id_norm = float(j) / max(n_jobs - 1, 1)

            job_available = float(state["job_available"][j]) / current_makespan
            machine_available = float(state["machine_available"][m]) / current_makespan

            tokens[op_id] = np.array(
                [
                    job_id_norm,
                    op_position,
                    machine_id_norm,
                    p / max_duration,
                    scheduled,
                    ready,
                    predecessor_done,
                    successor_exists,
                    job_progress,
                    job_available,
                    machine_available,
                    float(k == 0),
                    float(k == n_machines - 1),
                    float(state["job_next_op"][j] == k),
                    float(state["job_next_op"][j] > k),
                    1.0,
                ],
                dtype=np.float32,
            )

            if op_id in ready_ops:
                mask[op_id] = True

    return tokens, mask