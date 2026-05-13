import numpy as np 
import utils.constants as const
# Plot everything 

class HeuristicAgent:
  def __init__(self, env):
    self.env = env

  def get_action(self, state):
    """
    Hybrid Heuristic: Combines keyword detection, rate-limiting, 
    and threat-level estimation.
    """
    
    auth_log = state['auth_log']
    web_log = state['web_log']
    
   
    action = const.PASS_ACTION


    
    critical_signatures = ["jndi:ldap", "package.loadlib", "luaopen_io"]
    if any(sig in web_log.lower() for sig in critical_signatures):
      return const.BLOCK_ACTION

    
    failed_attempts = auth_log.lower().count("failed")
    
    if failed_attempts >= 5:
      action = const.BLOCK_ACTION

    elif failed_attempts >= 3:
      action = const.THROTTLE_ACTION

    elif failed_attempts >= 1:
      action = const.ALERT_ACTION

    

   
    if "success" in auth_log.lower() and failed_attempts == 0:
      action = const.UNBLOCK_ACTION


    return action