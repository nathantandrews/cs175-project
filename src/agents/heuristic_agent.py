import gymnasium as gym
import numpy as np
import security_gym
import constants as const

env = gym.make("SecurityLogStream-v1", db_path=const.BRUTE_7D_FILEPATH)
obs, info = env.reset()

# obs is a dict of text channels + system stats
print(obs["auth_log"][:200])   # Raw auth log lines
print(obs["system_stats"])     # [load_avg, mem_used, disk_used]

while True:
    # Choose an action
    action = {
        "action": 0,  # pass (monitor only)
        "risk_score": np.array([0.0], dtype=np.float32),
    }

    obs, reward, terminated, truncated, info = env.step(action)

    # Ground truth (for evaluation, not visible to agent)
    gt = info["ground_truth"]
    print(f"{info['timestamp']} | malicious={gt['is_malicious']} | "
          f"risk={gt['true_risk']:.1f} | reward={reward:.2f}")

    if truncated:  # End of data
        break
