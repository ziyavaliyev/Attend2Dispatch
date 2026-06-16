import numpy as np
import gymnasium as gym
import torch
import matplotlib
import matplotlib.pyplot as plt
if not hasattr(plt.cm, "get_cmap"):
    plt.cm.get_cmap = matplotlib.colormaps.get_cmap
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_rl.utils import clb

def make_graph_jsp_env(instances, cfg, seed, encoder=None, latent_dim=None, device="cpu", sample_latent=False,):
    env = DisjunctiveGraphJspEnv(
        jps_instance=instances[0],
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
        reward_function="trivial",
        reward_function_parameters={"scaling_divisor": 1000.0},
    )

    env = InstanceSamplerWrapper(env, instances, seed)
    env = ObservationWrapper(env, instances[0], device=device)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class InstanceSamplerWrapper(gym.Wrapper):
    def __init__(self, env, instances, seed=None):
        super().__init__(env)
        self.instances = [np.asarray(x, dtype=np.int64) for x in instances]
        self.rng = np.random.default_rng(seed)
        self.current_instance = None

    def reset(self, **kwargs):
        idx = int(self.rng.integers(0, len(self.instances)))

        self.current_instance = self.instances[idx]
        self.unwrapped.load_instance(self.current_instance)

        return self.env.reset(**kwargs)

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, instance, device="cpu"):
        super().__init__(env)

        T = instance.shape[1] * instance.shape[2]
        self.instance = instance
        self.device = torch.device(device)
    
        obs_dim = T+32
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(T, obs_dim),
            dtype=np.float32,
        )
        self.state = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(self.env, "current_instance"):
            self.instance = self.env.current_instance
        n_jobs = self.instance.shape[1]
        n_machines = self.instance.shape[2]
        self.state = {
            "job_next_op": np.zeros(n_jobs, dtype=np.int64),
            "machine_available": np.zeros(n_machines, dtype=np.int64),
            "job_available": np.zeros(n_jobs, dtype=np.int64),
            "scheduled": np.zeros(n_jobs * n_machines, dtype=np.bool_),
            "time": 0}
        return self.observation(obs), info

    def _update_state(self, op_id):
        n_machines = self.instance.shape[2]
        job_id = op_id // n_machines
        op_pos = op_id % n_machines
        machine_order = self.instance[0]
        durations = self.instance[1]
        machine_id = int(machine_order[job_id, op_pos])
        duration = int(durations[job_id, op_pos])
        start = max(
            int(self.state["job_available"][job_id]),
            int(self.state["machine_available"][machine_id]),
        )
        finish = start + duration
        self.state["job_next_op"][job_id] += 1
        self.state["job_available"][job_id] = finish
        self.state["machine_available"][machine_id] = finish
        self.state["scheduled"][op_id] = True
        self.state["time"] = finish
    
    def step(self, action):
        old_ms = int(np.max(self.state["machine_available"]))

        obs, graph_reward, terminated, truncated, info = self.env.step(action)

        self._update_state(action)

        new_ms = int(np.max(self.state["machine_available"]))

        reward = -(new_ms - old_ms) / 1000.0

        if terminated:
            reward += -new_ms / 1000.0
            info["makespan"] = new_ms

        return self.observation(obs), float(reward), terminated, truncated, info
    
    def observation(self, obs):
        state = self.state
        T = obs.shape[0]
        n_jobs = self.instance.shape[1]
        n_machines = self.instance.shape[2]

        machine_order = self.instance[0]
        proc_times = self.instance[1].astype(np.float32)

        max_duration = max(float(proc_times.max()), 1.0)
        total_work = max(float(proc_times.sum()), 1.0)
        job_total_work = proc_times.sum(axis=1)
        machine_total_work = np.zeros(n_machines, dtype=np.float32)

        for j in range(n_jobs):
            for k in range(n_machines):
                machine_total_work[int(machine_order[j, k])] += proc_times[j, k]

        max_job_work = max(float(job_total_work.max()), 1.0)
        max_machine_work = max(float(machine_total_work.max()), 1.0)
        job_prefix = np.cumsum(proc_times, axis=1)
        A = obs[:, :T]
        X = obs[:, T:]
        clb_values = clb(A, X).reshape(-1)

        valid_mask = self.env.unwrapped.valid_action_mask()

        current_makespan = max(float(np.max(state["machine_available"])), 1.0)
        time_scale = max(current_makespan, max_job_work, max_machine_work, 1.0)

        tokens = np.zeros((T, 32), dtype=np.float32)

        for op_id in range(T):
            job_id = op_id // n_machines
            op_pos = op_id % n_machines
            machine_id = int(machine_order[job_id, op_pos])
            duration = float(proc_times[job_id, op_pos])

            scheduled = float(state["scheduled"][op_id])
            ready = float(valid_mask[op_id])

            predecessor_done = 1.0 if op_pos == 0 else float(state["scheduled"][op_id - 1])
            successor_exists = float(op_pos < n_machines - 1)

            job_progress = float(state["job_next_op"][job_id]) / n_machines
            op_position = float(op_pos) / max(n_machines - 1, 1)
            machine_id_norm = float(machine_id) / max(n_machines - 1, 1)
            job_id_norm = float(job_id) / max(n_jobs - 1, 1)

            job_available = float(state["job_available"][job_id]) / time_scale
            machine_available = float(state["machine_available"][machine_id]) / time_scale

            current_op_pointer = int(state["job_next_op"][job_id])

            remaining_job_work = float(proc_times[job_id, current_op_pointer:].sum()) if current_op_pointer < n_machines else 0.0
            remaining_ops = max(n_machines - current_op_pointer, 0)

            machine_remaining_work = 0.0
            machine_queue_count = 0
            for j in range(n_jobs):
                for k in range(n_machines):
                    oid = j * n_machines + k
                    if not state["scheduled"][oid] and int(machine_order[j, k]) == machine_id:
                        machine_remaining_work += float(proc_times[j, k])
                        machine_queue_count += 1

            earliest_start = max(float(state["job_available"][job_id]), float(state["machine_available"][machine_id]))
            finish_if_now = earliest_start + duration
            waiting_gap = abs(float(state["job_available"][job_id]) - float(state["machine_available"][machine_id]))

            tail_after = float(proc_times[job_id, op_pos + 1:].sum()) if op_pos + 1 < n_machines else 0.0
            critical_score = float(job_prefix[job_id, op_pos] + tail_after) / max_job_work
            avg_remaining_duration = remaining_job_work / max(float(remaining_ops), 1.0)

            is_bottleneck_machine = float(machine_total_work[machine_id] == machine_total_work.max())
            is_ready_and_short = float(ready and duration <= np.median(proc_times))
            is_ready_and_critical = float(ready and critical_score >= 0.75)

            tokens[op_id] = np.array([
                job_id_norm,
                op_position,
                machine_id_norm,
                duration / max_duration,

                scheduled,
                ready,
                predecessor_done,
                successor_exists,

                job_progress,
                job_available,
                machine_available,

                float(op_pos == 0),
                float(op_pos == n_machines - 1),

                float(current_op_pointer == op_pos),
                float(current_op_pointer > op_pos),

                1.0,

                remaining_job_work / max_job_work,
                float(remaining_ops) / n_machines,
                job_total_work[job_id] / max_job_work,
                remaining_job_work / max(job_total_work[job_id], 1.0),

                machine_total_work[machine_id] / max_machine_work,
                machine_remaining_work / max_machine_work,
                float(machine_queue_count) / T,

                earliest_start / time_scale,
                finish_if_now / time_scale,
                waiting_gap / time_scale,

                clb_values[op_id],
                tail_after / max_job_work,
                critical_score,

                avg_remaining_duration / max_duration,
                duration / max(remaining_job_work, 1.0),
                is_bottleneck_machine,
            ], dtype=np.float32)

        return np.concatenate([A, tokens], axis=1).astype(np.float32)
        