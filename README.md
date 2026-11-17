# LocoMimic

G1 humanoid motion imitation with custom TD-MPC2, SAC, and the Unitree PPO baseline.
The [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) submodule supplies the simulation tasks and PPO.

## Setup

Use Linux with an NVIDIA GPU and a CUDA-compatible driver for training.
Clone the repository and install the dependencies:

```bash
git clone --recurse-submodules https://github.com/dineshvarunshankar/LocoMimic.git
cd LocoMimic
conda create -n locomimic python=3.11
conda activate locomimic
git submodule update --init --recursive
python -m pip install -e ./unitree_rl_mjlab
python -m pip install -r requirements.txt
```

The pinned submodule installs `mjlab==1.2.0` and `mujoco-warp==3.5.0`.

The example motion is `data/lafan1_retargeted_npz/g1/walk1_subject1.npz`.
To convert another G1 CSV motion, use the submodule converter:

```bash
mkdir -p data/lafan1_retargeted_npz/g1
python unitree_rl_mjlab/scripts/csv_to_npz.py \
  --robot g1 \
  --input-file /path/to/motion.csv \
  --output-name "$PWD/data/lafan1_retargeted_npz/g1/motion.npz" \
  --input-fps 30 --output-fps 50
```

Set `motion_path` in the algorithm configuration to use the converted motion.

## Run

Run all commands from the repository root.

### TD-MPC2

```bash
python -m scripts.train.train_tdmpc2 \
  --config agents/configs/tdmpc2_config.yaml \
  --name tdmpc2_walk1 \
  --motion-file data/lafan1_retargeted_npz/g1/walk1_subject1.npz \
  --num_envs 512

python -m scripts.eval.eval_tdmpc2 \
  --config agents/configs/tdmpc2_config.yaml \
  --checkpoint models/tdmpc2_walk1/final.pt \
  --motion-file data/lafan1_retargeted_npz/g1/walk1_subject1.npz \
  --num-envs 1 --episodes 1
```

Add `--disable-wandb` to train without W&B.
Add `--video` to evaluation to save an MP4.
Use `--episode-length-s 20` to set the evaluation horizon.
The `--no-terminations` flag disables failure conditions and retains the episode timeout.

### SAC

```bash
python -m scripts.train.train_sac \
  --config agents/configs/sac_config.yaml \
  --name sac_walk1 \
  --task Unitree-G1-Tracking --num_envs 4096

python -m scripts.eval.eval_sac \
  --config agents/configs/sac_config.yaml \
  --ckpt models/sac_walk1/final.pt \
  --task Unitree-G1-Tracking --device cuda:0
```

SAC evaluation opens the native viewer.
Keep each `.rms.pt` normalization file beside its `.pt` checkpoint.
Use `--fast-sac` for both training and evaluation of the Fast SAC variant.
SAC `buffer_size` counts transitions per environment.

### PPO

```bash
python unitree_rl_mjlab/scripts/train.py Unitree-G1-Tracking \
  --motion-file data/lafan1_retargeted_npz/g1/walk1_subject1.npz \
  --env.scene.num-envs 4096

python unitree_rl_mjlab/scripts/play.py Unitree-G1-Tracking \
  --motion-file data/lafan1_retargeted_npz/g1/walk1_subject1.npz \
  --checkpoint-file /path/to/model.pt
```

PPO checkpoints are in `logs/rsl_rl/`.
SAC and TD-MPC2 checkpoints are in `models/<run_name>/`.

## References

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
