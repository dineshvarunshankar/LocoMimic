# LocoMimic

Tools for training and evaluating human motion imitation on the Unitree G1 humanoid.
Includes custom TD-MPC2, SAC, and the PPO baseline from [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).

## Setup

Use Linux with an NVIDIA GPU and a compatible CUDA driver.

```bash
git clone --recurse-submodules https://github.com/dineshvarunshankar/LocoMimic.git
cd LocoMimic
conda create -n locomimic python=3.11
conda activate locomimic
python -m pip install -e ./unitree_rl_mjlab
python -m pip install -r requirements.txt
```

## Usage

Run commands from the repository root.
SAC and TD-MPC2 use the configurations in `agents/configs/` and the included walking motion.

### Training

**TD-MPC2**

```bash
python -m scripts.train.train_tdmpc2 --name tdmpc2_walk1
```

**SAC**

```bash
python -m scripts.train.train_sac --name sac_walk1
```

**PPO**

```bash
python unitree_rl_mjlab/scripts/train.py Unitree-G1-Tracking \
  --motion-file data/lafan1_retargeted_npz/g1/walk1_subject1.npz \
  --env.scene.num-envs 4096
```

### Evaluation

**TD-MPC2**

```bash
python -m scripts.eval.eval_tdmpc2 \
  --checkpoint models/tdmpc2_walk1/final.pt --episodes 1
```

Add `--video` to save an MP4.

**SAC**

```bash
python -m scripts.eval.eval_sac \
  --ckpt models/sac_walk1/final.pt --device cuda:0
```

Keep the `.rms.pt` normalization file beside its checkpoint.
For Fast SAC, pass `--fast-sac` during both training and evaluation.

**PPO**

```bash
python unitree_rl_mjlab/scripts/play.py Unitree-G1-Tracking \
  --motion-file data/lafan1_retargeted_npz/g1/walk1_subject1.npz \
  --checkpoint-file /path/to/model.pt
```

SAC and TD-MPC2 save checkpoints in `models/<run_name>/`; PPO uses `logs/rsl_rl/`.

## Project Structure

```text
agents/
  sac/                       SAC and distributional critic
  tdmpc2/                    World model and MPC planner
  configs/                   Algorithm configurations
rl/                          Environment wrappers
scripts/
  train/                     Training entry points
  eval/                      Evaluation and video tools
unitree_rl_mjlab/             Simulation tasks and PPO baseline
data/lafan1_retargeted_npz/   Motion data
```

## References

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
