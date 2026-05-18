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
BRUTE_7D_FILEPATH = "data/exp_7d_brute.db"

