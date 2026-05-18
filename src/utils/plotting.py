import matplotlib.pyplot as plt

def plot_learning_curve(rewards, save_path):
  plt.figure(figsize=(12, 6))
  plt.plot(rewards, label="Reward per Episode")
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  plt.title("DQN Learning Curve")
  plt.legend()
  plt.grid()
  plt.savefig(save_path)
  plt.close()
  
