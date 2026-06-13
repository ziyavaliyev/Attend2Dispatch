import argparse
import os
import time
import random
import yaml
import json
from tqdm import trange
from jsp_rl.encoder import Encoder, VariationalEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import wandb

from jsp_rl.utils import generate_jsp_instance
from jsp_rl.rl_model import JSPActorCritic
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_rl.env_wrapper import (
    make_graph_jsp_env,
    ObservationWrapper,
)

def load_val_dataset(path):
    data = torch.load(path, weights_only=False)
    return data["instances"], data["summary"]

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_pretrained_encoder(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc_cfg = ckpt["config"]

    deg = ckpt.get("deg", None)
    if deg is not None:
        deg = deg.to(device)

    encoder_type = enc_cfg.get("model", "gae")  # your key: "model"

    EncoderClass = VariationalEncoder if encoder_type == "vgae" else Encoder

    encoder = EncoderClass(
        in_channels=enc_cfg["in_dim"],
        hidden_channels=enc_cfg["hidden_dim"],
        out_channels=enc_cfg["latent_dim"],
        gnn_type=enc_cfg["gnn_type"],
        deg=deg,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder, enc_cfg["latent_dim"]

def make_env(instances, cfg, seed, encoder=None, latent_dim=None, device="cpu"):
    def thunk():
        return make_graph_jsp_env(
            instances,
            cfg,
            seed,
            encoder=encoder,
            latent_dim=latent_dim,
            device=device,
            sample_latent=cfg["encoder"].get("sample_latent", False),
        )
    return thunk

def collect_masks(envs, device):
    masks = envs.call("valid_action_mask")
    masks = np.stack(masks, axis=0)
    return torch.tensor(masks, dtype=torch.bool, device=device)

def build_instances(cfg, split="train"):
    rng = random.Random(cfg["seed"] + (0 if split == "train" else 999))

    n = cfg["data"]["n_train_instances"] if split == "train" else cfg["data"]["n_val_instances"]

    return [
        generate_jsp_instance(
            n_jobs=cfg["data"]["n_jobs"],
            n_machines=cfg["data"]["n_machines"],
            min_duration=cfg["data"]["min_duration"],
            max_duration=cfg["data"]["max_duration"],
            rng=rng,
        )
        for _ in range(n)
    ]


@torch.no_grad()
def evaluate_rl_model(
    model,
    val_instances,
    baseline_summary,
    cfg,
    device,
    encoder=None,
    latent_dim=None,
):
    model.eval()

    makespans = []

    for instance in val_instances:
        result = rollout_policy_from_ac(
            model,
            instance,
            cfg,
            device,
            encoder=encoder,
            latent_dim=latent_dim,
        )

        makespans.append(result["makespan"])

    mean_makespan = float(np.mean(makespans))
    std_makespan = float(np.std(makespans))

    metrics = {
        "mean_makespan": mean_makespan,
        "std_makespan": std_makespan,
    }

    if "ortools" in baseline_summary:
        ortools_mean = baseline_summary["ortools"]["mean"]

        optimality_gap = (
            (mean_makespan - ortools_mean)
            / ortools_mean
            * 100.0
        )

        metrics["optimality_gap_percent"] = float(optimality_gap)

    return metrics

@torch.no_grad()
def rollout_policy_from_ac(model, instance, cfg, device, encoder=None, latent_dim=None):
    model.eval()

    env = DisjunctiveGraphJspEnv(
        jps_instance=instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
        reward_function="zero"
    )

    env = ObservationWrapper(
        env,
        instance,
        obs_mode=cfg["observation"]["mode"],
        encoder=encoder,
        latent_dim=latent_dim,
        device=device,
        sample_latent=cfg["encoder"].get("sample_latent", False))

    obs, _ = env.reset()

    done = False
    truncated = False

    actions = []

    while not (done or truncated):

        mask = env.unwrapped.valid_action_mask()

        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        mask_t = torch.tensor(
            mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)

        logits, _ = model.get_logits_and_value(
            obs_t,
            mask_t,
        )

        action = int(torch.argmax(logits, dim=1).item())

        obs, reward, done, truncated, info = env.step(action)

        actions.append(action)

    return {
        "makespan": info["makespan"],
        "actions": actions,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_10x10.yaml")
    parser.add_argument("--representation", type=str, default=None, help="Override representation for sweeps.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.representation is not None:
        rep = args.representation

        if rep in ["handcrafted", "raw_graph", "graph_features"]:
            cfg["observation"]["mode"] = rep

        else:
            cfg["observation"]["mode"] = "encoder"
            cfg["encoder"]["path"] = rep

        rep_name = os.path.splitext(os.path.basename(args.representation))[0]

        if rep_name in ["handcrafted", "raw_graph", "graph_features"]:
            cfg["logging"]["run_name"] = rep_name
        else:
            cfg["logging"]["run_name"] = rep_name

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo = cfg["ppo"]

    num_envs = int(ppo["num_envs"])
    num_steps = int(ppo["num_steps"])
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // int(ppo["num_minibatches"])
    num_iterations = int(ppo["total_timesteps"]) // batch_size

    run_name = cfg["logging"]["run_name"] + f"__{seed}__{int(time.time())}"

    if cfg["logging"]["use_wandb"]:
        wandb.init(
            project=cfg["logging"]["project"],
            name=run_name,
            config=cfg,
            sync_tensorboard=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    train_instances = build_instances(cfg, split="train")
    
    val_instances, baseline_summary = load_val_dataset(cfg["data"]["val_dataset_path"])

    print("Validation baselines:")
    for name, metrics in baseline_summary.items():
        print(f"{name}: mean={metrics['mean']:.2f}")

    if cfg["logging"]["use_wandb"]:
        baseline_log = {}

        for name, metrics in baseline_summary.items():
            baseline_log[f"baseline/{name}_mean_makespan"] = metrics["mean"]

        wandb.log(baseline_log, step=0)

    encoder = None
    latent_dim = None
    if cfg["observation"]["mode"] == "encoder":
        encoder, latent_dim = load_pretrained_encoder(cfg["encoder"]["path"], device)

    envs = gym.vector.SyncVectorEnv([
        make_env(
            train_instances,
            cfg,
            seed + i,
            encoder=encoder,
            latent_dim=latent_dim,
            device=device,
        )
        for i in range(num_envs)
    ])

    n_tokens = cfg["data"]["n_jobs"] * cfg["data"]["n_machines"]

    token_dim = envs.single_observation_space.shape[-1]
    agent = JSPActorCritic(
        token_dim=token_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        n_tokens=n_tokens,
    ).to(device)

    """bc_ckpt = cfg["model"].get("bc_checkpoint", None)
    if bc_ckpt and os.path.exists(bc_ckpt):
        agent.load_bc_actor(bc_ckpt)"""

    optimizer = optim.Adam(agent.parameters(), lr=ppo["learning_rate"], eps=1e-5)

    obs = torch.zeros((num_steps, num_envs, n_tokens, token_dim), device=device)
    actions = torch.zeros((num_steps, num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    masks_buf = torch.zeros((num_steps, num_envs, n_tokens), device=device, dtype=torch.bool)

    global_step = 0
    start_time = time.time()

    last_eval_step = 0
    eval_every = int(cfg["logging"]["eval_every"])

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    
    next_done = torch.zeros(num_envs, device=device)

    best_val_makespan = float("inf")
    best_optimality_gap = float("inf")
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)

    pbar = trange(1, num_iterations + 1, desc="Training")
    for iteration in pbar:
        if ppo["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * ppo["learning_rate"]

        for step in range(num_steps):
            global_step += num_envs

            obs[step] = next_obs
            dones[step] = next_done

            mask = collect_masks(envs, device)
            masks_buf[step] = mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, mask)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                final_infos = infos["final_info"]
                for info in final_infos:
                    if info and "makespan" in info:
                        writer.add_scalar("charts/train_episode_makespan", info["makespan"], global_step)
                        if cfg["logging"]["use_wandb"]:
                            wandb.log({"charts/train_episode_makespan": info["makespan"]}, step=global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + ppo["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + ppo["gamma"] * ppo["gae_lambda"] * nextnonterminal * lastgaelam

            returns = advantages + values

        b_obs = obs.reshape((-1, n_tokens, token_dim))
        b_masks = masks_buf.reshape((-1, n_tokens))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(ppo["update_epochs"]):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_masks[mb_inds],
                    b_actions[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > ppo["clip_coef"]).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if ppo["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - ppo["clip_coef"],
                    1 + ppo["clip_coef"],
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)

                if ppo["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -ppo["clip_coef"],
                        ppo["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ppo["ent_coef"] * entropy_loss + ppo["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo["max_grad_norm"])
                optimizer.step()

            if ppo["target_kl"] is not None and approx_kl > ppo["target_kl"]:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        pbar.set_postfix({
            "pg": f"{pg_loss.item():.3f}",
            "v": f"{v_loss.item():.3f}",
            "ent": f"{entropy_loss.item():.3f}",
            "mk": f"{best_val_makespan:.1f}",
        })

        if global_step - last_eval_step >= eval_every:
            last_eval_step = global_step
            val_metrics = evaluate_rl_model(agent, val_instances, baseline_summary, cfg, device, encoder=encoder, latent_dim=latent_dim)
            """if "optimality_gap_percent" in val_metrics:
                best_optimality_gap = min(best_optimality_gap, val_metrics["optimality_gap_percent"])"""
            writer.add_scalar("val/mean_makespan", val_metrics["mean_makespan"], global_step)
            writer.add_scalar("val/std_makespan", val_metrics["std_makespan"], global_step)
            if "optimality_gap_percent" in val_metrics:
                writer.add_scalar(
                    "val/optimality_gap_percent",
                    val_metrics["optimality_gap_percent"],
                    global_step)

            if cfg["logging"]["use_wandb"]:
                log_dict = {
                    "val/mean_makespan": val_metrics["mean_makespan"],
                    "val/optimality_gap_percent": val_metrics["optimality_gap_percent"],
                    "val/best_optimality_gap_percent": best_optimality_gap,
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                }
                wandb.log(log_dict, step=global_step)

            print(
                f"step={global_step} "
                f"val_mean_makespan={val_metrics['mean_makespan']:.2f} "
                f"gap={val_metrics['optimality_gap_percent']:.2f}% "
                f"SPS={int(global_step / (time.time() - start_time))}"
            )

            current_gap = val_metrics["optimality_gap_percent"]
            if current_gap < best_optimality_gap:
                best_optimality_gap = current_gap
                best_val_makespan = val_metrics["mean_makespan"]

                torch.save(
                    agent.state_dict(),
                    f"runs/{run_name}/checkpoints/best_rl.pt"
                )

                with open(
                    f"runs/{run_name}/checkpoints/best_metrics.json",
                    "w"
                ) as f:
                    json.dump(val_metrics, f, indent=2)

    final_metrics = evaluate_rl_model(
        agent,
        val_instances,
        baseline_summary,
        cfg,
        device,
        encoder=encoder,
        latent_dim=latent_dim,
    )

    writer.add_scalar("final/mean_makespan", final_metrics["mean_makespan"], global_step)
    writer.add_scalar("final/std_makespan", final_metrics["std_makespan"], global_step)

    if "optimality_gap_percent" in final_metrics:
        writer.add_scalar(
            "final/optimality_gap_percent",
            final_metrics["optimality_gap_percent"],
            global_step,
        )

    if cfg["logging"]["use_wandb"]:
        wandb.log(
            {
                "final/mean_makespan": final_metrics["mean_makespan"],
                "final/optimality_gap_percent": final_metrics.get("optimality_gap_percent"),
                "val/best_optimality_gap_percent": best_optimality_gap,
            },
            step=global_step,
        )

    torch.save(agent.state_dict(), f"runs/{run_name}/last_rl.pt")

    envs.close()
    writer.close()

    if cfg["logging"]["use_wandb"]:

        best_model_path = f"runs/{run_name}/checkpoints/best_rl.pt"
        best_metrics_path = f"runs/{run_name}/checkpoints/best_metrics.json"

        if os.path.isfile(best_model_path):

            artifact = wandb.Artifact(
                name=f"{run_name}-best-model",
                type="model",
            )

            artifact.add_file(best_model_path)

            if os.path.isfile(best_metrics_path):
                artifact.add_file(best_metrics_path)

            wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()