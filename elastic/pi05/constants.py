"""pi0.5 backend defaults: reward shaping, replay layout and online-training cadence.

Strides are in flow time [0, 1]; ``DIFFUSION_T_MAX`` converts integer step budgets.
"""

PAUSE_ACTION_THRESHOLD = 0.95
DIFFUSION_T_MAX = 100.0

# Terminal bonus and verifier bias are calibrated to the V-GPS score scale on LIBERO replays.
LATENCY_PENALTY = 0.1
PARALLEL_PENALTY = 0.1
TERMINAL_BONUS = 2
VERIFIER_BIAS = 2
NO_FINISH_PENALTY = -30.0

BATCH_SIZE = 256
OFFLINE_BATCH_FRAC = 0.2
OFFLINE_PRETRAIN_STEPS = 5000
ONLINE_UPDATES_PER_STEP = 1
ONLINE_MIN_TRANSITIONS = 0
STRIDE_REPLAY_CAPACITY = 200_000
# Online steps between checkpoints; <= 0 saves only best and at exit.
STRIDE_CHECKPOINT_EVERY = 100
STRIDE_USE_EXTENDED_OBS = False
DEFAULT_STRIDE_WANDB_PROJECT = "elastic_pi05"

RANDOM_STEP_OPTIONS = "5,10,20,30,50"
STRIDE_BUCKETS = [5, 10, 20, 30, 40, 50, 75, 100]
STRIDE_BUCKET_WEIGHTS = [0.30, 0.30, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02]

K_SUBSAMPLE_DRAWS = 2
K_SUBSAMPLE_ADD_COUNTERFACTUALS = True
# One row per infer with every sample paused; reward = no_finish_penalty.
ALL_STOPPED_COUNTERFACTUALS = True
# Second pass: re-denoise the chunk at a fixed stride drawn around the episode's average
# stride (scaled by [MIN, MAX]); its mean V-GPS score is the advantage baseline V(s).
SECOND_PASS_COUNTERFACTUALS = True
SECOND_PASS_EVERY = 4
CF_STRIDE_SCALE_MIN = 0.5
CF_STRIDE_SCALE_MAX = 1.5
CF_INITIAL_AVG_STRIDE = 1.0 / 10.0

# reward_components: [verifier_quality, latency_steps, parallel_norm_excess, robot_episode_success].
# The last column is an analysis label only; replays without it load it as 1.0.
REPLAY_REWARD_COMPONENT_DIM = 4
# stride_infer_meta: adaptive_steps, avg_active_samples, finished_fraction, num_transitions,
# baseline_num_steps, selected_finished, copied onto every row from that infer.
STRIDE_INFER_META_DIM = 6
# traj_info[:, 1] markers for synthetic rows (real rows hold the infer index within the episode).
REPLAY_STRIDE_INFER_IDX_K_SUBSAMPLE = -2
REPLAY_STRIDE_INFER_IDX_SECOND_PASS = -3
REPLAY_STRIDE_INFER_IDX_ALL_STOPPED = -4
SCORE_ENCODING_ADVANTAGE = "advantage"
SCORE_ENCODING_RAW_Q = "raw_q"
