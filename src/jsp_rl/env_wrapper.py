import random
import numpy as np
import gymnasium as gym
import torch
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_rl.utils import build_graph_node_features

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
    env = ObservationWrapper(env, instances[0], obs_mode=cfg["observation"]["mode"], encoder=encoder, latent_dim=latent_dim, device=device, sample_latent=sample_latent)
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
    def __init__(self, env, instance, obs_mode="handcrafted", encoder=None, latent_dim=None, device="cpu", sample_latent=False):
        super().__init__(env)

        T = instance.shape[1] * instance.shape[2]
        self.sample_latent = sample_latent
        self.instance = instance
        self.encoder = encoder
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.obs_mode = obs_mode
        if self.obs_mode == "encoder":
            if encoder is None:
                raise ValueError("obs_mode='encoder' requires an encoder.")
            if latent_dim is None:
                raise ValueError("obs_mode='encoder' requires latent_dim.")
            self.encoder.to(self.device)
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            obs_dim = latent_dim
        elif self.obs_mode == "raw_graph":
            obs_dim = T + self.instance.shape[2] + 1
        elif self.obs_mode == "graph_features":
            obs_dim = T + self.instance.shape[2] + 2
        elif self.obs_mode == "handcrafted":
            obs_dim = 16
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(T, obs_dim),
            dtype=np.float32,
        )
        self.state = None
    
    """def _edge_index_from_obs(self, obs):
        T = obs.shape[0]
        A = obs[:, :T]
        src, dst = np.nonzero(A > 0)

        if src.size == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        return torch.tensor(
            np.stack([src, dst], axis=0),
            dtype=torch.long,
            device=self.device,
        )"""

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

    def _encode_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        n_machines = self.instance.shape[2]
        A, x_np = build_graph_node_features(obs=obs, scheduled=self.state["scheduled"], n_machines=n_machines)
        src, dst = np.nonzero(A > 0)
        edge_index = (
            torch.tensor(
                np.stack([src, dst], axis=0),
                dtype=torch.long,
                device=self.device,
            )
            if src.size
            else torch.empty((2, 0), dtype=torch.long, device=self.device)
        )

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z = self.encoder(x, edge_index)
            if isinstance(z, tuple):
                mu, logstd = z
                if self.sample_latent:
                    std = torch.exp(logstd)
                    z = mu + torch.randn_like(std) * std
                else:
                    z = mu

        return z.cpu().numpy().astype(np.float32)
    
    def _raw_graph_features(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs#np.random.rand(*obs.shape).astype(np.float32)
    
    def _graph_features(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        n_machines = self.instance.shape[2]
        A, x_np = build_graph_node_features(
            obs=obs,
            scheduled=self.state["scheduled"],
            n_machines=n_machines,
        )
        return np.concatenate([A, x_np], axis=1).astype(np.float32)

    def _handcrafted_features(self, obs):
        state = self.state
        T = obs.shape[0]
        n_jobs = self.instance.shape[1]
        n_machines = self.instance.shape[2]
        machine_order = self.instance[0]
        proc_times = self.instance[1]
        max_duration = max(float(proc_times.max()), 1.0)
        tokens = np.zeros((T, 16), dtype=np.float32)
        valid_mask = self.env.unwrapped.valid_action_mask()
        current_makespan = max(
            float(np.max(state["machine_available"])),
            1.0,
        )
        for op_id in range(T):
            job_id = op_id // n_machines
            op_pos = op_id % n_machines
            machine_id = int(machine_order[job_id, op_pos])
            scheduled = float(state["scheduled"][op_id])
            ready = float(valid_mask[op_id])
            predecessor_done = 1.0
            if op_pos > 0:
                predecessor_done = float(
                    state["scheduled"][op_id - 1]
                )

            successor_exists = float(
                op_pos < n_machines - 1
            )

            job_progress = (
                float(state["job_next_op"][job_id])
                / n_machines
            )

            op_position = (
                float(op_pos)
                / max(n_machines - 1, 1)
            )

            machine_id_norm = (
                float(machine_id)
                / max(n_machines - 1, 1)
            )

            job_id_norm = (
                float(job_id)
                / max(n_jobs - 1, 1)
            )

            job_available = (
                float(state["job_available"][job_id])
                / current_makespan
            )

            machine_available = (
                float(state["machine_available"][machine_id])
                / current_makespan
            )

            current_op_pointer = int(
                state["job_next_op"][job_id]
            )

            tokens[op_id] = np.array(
                [
                    job_id_norm,
                    op_position,
                    machine_id_norm,
                    float(proc_times[job_id, op_pos]) / max_duration,

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
                ],
                dtype=np.float32,
            )
        return tokens
    
    def observation(self, obs):
        if self.obs_mode == "encoder":
            return self._encode_obs(obs)

        if self.obs_mode == "raw_graph":
            return self._raw_graph_features(obs)

        if self.obs_mode == "graph_features":
            return self._graph_features(obs)

        if self.obs_mode == "handcrafted":
            return self._handcrafted_features(obs)

        raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
        