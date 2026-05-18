import numpy as np

# Does nothing, allows event to pass through unmodified
PASS_ACTION = {"action": 0, "risk_score": np.array([0.0], dtype=np.float32)}

# Block the current event's source IP (100% drop)
BLOCK_ACTION = {"action": 3, "risk_score": np.array([8.0], dtype=np.float32)}

# Throttle (90% drop rate)
THROTTLE_ACTION = {"action": 2, "risk_score": np.array([5.0], dtype=np.float32)}

# Alert with high risk estimate
ALERT_ACTION = {"action": 1, "risk_score": np.array([7.0], dtype=np.float32)}

# Undo a block (correct false positive)
UNBLOCK_ACTION = {"action": 4, "risk_score": np.array([1.0], dtype=np.float32)}

# Quarantine server (blocks all network events)
ISOLATE_ACTION = {"action": 5, "risk_score": np.array([10.0], dtype=np.float32)}

# Dataset paths
BENIGN_V3_FILEPATH = "data/benign_v3.db"
CAMPAIGNS_V2_FILEPATH = "data/campaigns_v2.db"
EXP01_90D_FILEPATH = "data/exp01_90d.db"
EXP_30D_HEAVY_FILEPATH = "data/exp_30d_heavy.db"
EXP_365D_REALISTIC_FILEPATH = "data/exp_365d_realistic.db"
EXP_7D_BRUTE_FILEPATH = "data/exp_7d_brute.db"

DATASETS = {
    "benign_v3": BENIGN_V3_FILEPATH,
    "campaigns_v2": CAMPAIGNS_V2_FILEPATH,
    "exp01_90d": EXP01_90D_FILEPATH,
    "exp_30d_heavy": EXP_30D_HEAVY_FILEPATH,
    "exp_365d_realistic": EXP_365D_REALISTIC_FILEPATH,
    "exp_7d_brute": EXP_7D_BRUTE_FILEPATH,
}
