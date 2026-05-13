import numpy as np 
import constants as const


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
    risk_score = 0.0

    
    critical_signatures = ["jndi:ldap", "package.loadlib", "luaopen_io"]
    if any(sig in web_log.lower() for sig in critical_signatures):
        return {
            "action": const.BLOCK_ACTION,
            "risk_score": 10.0  # Maximum certainty
        }

    
    failed_attempts = auth_log.lower().count("failed")
    
    if failed_attempts >= 5:
        action = const.BLOCK_ACTION
        risk_score = 9.0
    elif failed_attempts >= 3:
        action = const.THROTTLE_ACTION
        risk_score = 6.5
    elif failed_attempts >= 1:
        action = const.ALERT_ACTION
        risk_score = 4.0
    

   
    if "success" in auth_log.lower() and failed_attempts == 0:
        action = const.UNBLOCK_ACTION
        risk_score = 1.0

    return {
        "action": action,
        "risk_score": risk_score
    }