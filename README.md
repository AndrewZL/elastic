# ELASTIC

Adaptive stride meta-policy on a frozen generative policy: Diffusion Policy (PushT, robomimic) and pi0.5 (LIBERO-10). Both backends are unmodified upstream submodules and ELASTIC works on top of a pretrained frozen checkpoint.

A frozen generative policy denoises S candidate action chunks in parallel. ELASTIC's stride actor decides, per candidate and per denoising step, how far to jump (or to stop).  A verifier scores the finished candidates and the best one is executed. 

The actor is trained with SAC on the team reward: the chosen candidate's verifier advantage minus latency (sequential denoise steps) and parallel (active samples) penalties, with counterfactual replay rows for strides it did not take.

## Install

```bash
git submodule update --init --recursive
conda activate robodiff
pip install -e .
```

Put frozen DP checkpoints, verifier weights, replay buffers, and robomimic hdf5s under `data/`, or point `$ELASTIC_DATA` at them. 

```
ckpts/{task}_lowdim.ckpt
datasets/{task}/low_dim.hdf5
```

#### pi0.5

```bash
docker build -t openpi_server -f backends/openpi/scripts/docker/serve_policy.Dockerfile backends/openpi
docker build -t elastic_pi05_server - < docker/pi05/server.Dockerfile
docker build -t elastic_libero -f docker/pi05/libero.Dockerfile .
```

Put the V-GPS checkpoint at `data/verifiers/vgps_libero10r/`.


## Run

#### Diffusion Policy

```bash
python -m elastic.dp.training.train --config configs/jobs/pusht_obs/pen_0p2_0p1.yaml
python scripts/dp/evaluate.py --config configs/eval/pusht_obs.yaml \
  --stride_checkpoint results/pusht_obs/pen_0p2_0p1/checkpoints/best_eval_stride_policy.pt \
  --output_dir results/eval/pusht_obs/pen_0p2_0p1
```
Flags passed to `scripts/dp/evaluate.py` override `--config`. 

---

Regenerate offline replay or verifier training data:

```bash
python -m elastic.dp.training.collect buffer --config configs/jobs/pusht_obs/pen_0p2_0p1.yaml
python -m elastic.dp.training.collect verifier --checkpoint data/ckpts/pusht_lowdim.ckpt --obstacle \
  --output_path data/verifier_datasets/pusht_obs.pkl --ref_output_path data/verifier_datasets/pusht_obs_ref.pkl
```

Retrain a verifier:

```bash
python scripts/dp/train_verifier.py pusht          
python scripts/dp/train_verifier.py square
```

Then modify data paths in config or put them in
```
verifiers/{task}.pth
buffers/{pusht_S4,pusht_obs_S4,can_paired_S4,can_reverse_S8,square_mh_S4,square_ph_S4}.pkl
verifier_datasets/{task}.pkl
verifier_datasets/{task}_ref.pkl
```
Checkpoints / data coming soon!

#### Pi0.5

```bash
python scripts/pi05/train.py --collect_demos --output_dir data/libero/runs/my_run
python scripts/pi05/eval.py --paper-baselines \
  --stride-run elastic=data/libero/runs/my_run/checkpoints/last_stride_learner.pt
```

To use V-GPS as the verifier, follow training instructions in [V-GPS](https://github.com/nakamotoo/v-gps).

