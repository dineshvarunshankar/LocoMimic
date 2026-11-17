import argparse
import math

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from agents.configs.config_loader import load_tdmpc2_config
from agents.tdmpc2.tdmpc2_agent import TDMPC2Agent
from rl.final_observation_env import FinalObservationEnv
from rl.tdmpc2_env_wrapper import TDMPC2VecEnvWrapper


def _configure_evaluation_horizon(env_cfg, episode_length_s: float, no_terminations: bool):
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("--episode-length-s must be finite and positive")
    env_cfg.episode_length_s = episode_length_s
    if no_terminations:
        env_cfg.terminations = {
            name: term for name, term in env_cfg.terminations.items() if term.time_out
        }
    # Play configs can remove this term or use a practically infinite horizon.
    env_cfg.terminations["time_out"] = TerminationTermCfg(func=time_out, time_out=True)


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="agents/configs/tdmpc2_config.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-length-s", type=float, default=None,
                        help="Episode duration; defaults to the training task horizon")
    parser.add_argument("--video", action="store_true", help="Record an MP4 video of the evaluation")
    parser.add_argument("--no-terminations", action="store_true", help="Disable early terminations so the episode plays out")
    args = parser.parse_args()

    cfg = load_tdmpc2_config(args.config)
    if args.task is not None:
        cfg.task = args.task
    if args.motion_file is not None:
        cfg.motion_path = args.motion_file
    cfg.num_envs = max(1, args.num_envs)
    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"

    configure_torch_backends()
    env_cfg = load_env_cfg(cfg.task, play=True)
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = cfg.motion_path
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.observations["actor"].concatenate_terms = False

    episode_length_s = args.episode_length_s
    if episode_length_s is None:
        episode_length_s = float(load_env_cfg(cfg.task).episode_length_s)
    _configure_evaluation_horizon(env_cfg, episode_length_s, args.no_terminations)
    if args.no_terminations:
        print("[INFO] Failure terminations disabled; episode timeouts remain enabled.")

    render_mode = "rgb_array" if args.video else None
    env = FinalObservationEnv(cfg=env_cfg, device=cfg.device, render_mode=render_mode)
    
    if args.video:
        import os
        from mjlab.utils.wrappers import VideoRecorder
        video_dir = os.path.join(os.path.dirname(args.checkpoint), "videos")
        print(f"[INFO] Video recording enabled. Saving to {video_dir}")
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=1000,
            disable_logger=True,
        )

    env = TDMPC2VecEnvWrapper(env, exogenous_terms=tuple(cfg.exogenous_terms), clip_actions=None)

    agent = TDMPC2Agent(
        endog_obs_dim=env.endog_obs_dim,
        exog_obs_dim=env.exog_obs_dim,
        act_dim=env.act_dim,
        config=cfg,
    )
    agent.load(args.checkpoint)

    endog_obs, exog_obs, _ = env.reset()
    episode_returns = torch.zeros(cfg.num_envs, device=cfg.device)
    completed: list[float] = []

    while len(completed) < args.episodes:
        planner_env_ids = None
        if cfg.mpc_eval:
            planner_env_ids = torch.arange(cfg.num_envs, device=cfg.device)
        exog_plan_seq = env.get_exog_plan_sequence(cfg.horizon)

        action = agent.act(
            endog_obs=endog_obs,
            exog_obs=exog_obs,
            exog_plan_seq=exog_plan_seq,
            deterministic=True,
            planner_env_ids=planner_env_ids,
        )
        next_endog, next_exog, reward, terminated, time_out, _ = env.step(action)
        episode_returns += reward

        done_any = terminated | time_out
        if done_any.any():
            finished = done_any.nonzero(as_tuple=False).squeeze(-1)
            for idx in finished:
                completed.append(float(episode_returns[idx].item()))
                if len(completed) >= args.episodes:
                    break
            episode_returns[finished] = 0.0
            agent.reset_planner(finished)

        endog_obs = next_endog
        exog_obs = next_exog

    env.close()
    mean_return = sum(completed) / max(len(completed), 1)
    print(f"Evaluated {len(completed)} episodes | mean return: {mean_return:.3f}")


if __name__ == "__main__":
    evaluate()
