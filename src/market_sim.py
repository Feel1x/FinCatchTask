import random
from collections import Counter

"""
MarketEnvironment simulates market conditions.
Uses an MLP (LLM sim) and an Agent to drive interactions,
computing rewards based on digit matching between the LLM output and the agent action.
"""

class MarketEnvironment:
    def __init__(self, mlp, agent):
        # assign mlp & agent, then clear counters
        self.mlp = mlp
        self.agent = agent
        self.reset()
    
    def reset(self):
        # clear interaction count and previous match state
        self.interaction_count = 0
        self.prev_state = None  # Tidigare "state" (antal matchande siffror)
    
    def step(self, market_input: int):
        # update count, get LLM output, agent action, and compute match state
        self.interaction_count += 1
        llm_output = self.mlp.analyze(market_input)
        action, log_prob = self.agent.get_action(llm_output)
        state = self._count_matches(llm_output, action)
        
        # finish if state is unchanged, or after 10 interactions, plus added being 3 as a condition!
        done = (self.prev_state is not None and state == self.prev_state and state == 3) or (self.interaction_count >= 10)
        reward = (100 / self.interaction_count) if (done and state == 3) else 0
        self.prev_state = state
        
        # set next input to state if 1<=state<=9; else default to 1 or random
        next_input = state if 1 <= state <= 9 else 1 ###random.randint(10, 100)
        return llm_output, action, state, reward, done, next_input, log_prob

    def _count_matches(self, llm_output, action):
        # count matching digits between LLM output and agent's 3-digit action similar way as calc_reward()
        llm_str = str(llm_output)
        action_str = str(action).zfill(3)
        return sum(min(llm_str.count(d), action_str.count(d)) for d in set(action_str))


if __name__ == "__main__":
    from mlp_for_llm_sim import MLP
    from agent import AgentPolicyNetwork
    from market_sim import MarketEnvironment

    mlp = MLP()
    agent = AgentPolicyNetwork()
    env = MarketEnvironment(mlp, agent)

    """
    This runs simulation starting with market_input = 1.
    At each step, the environment processes the input, gets LLM output, agent action,
    and computes the match count. The simulation ends when a matching state of 3 repeats or after 10 steps.
    Each step's details and the final reward are printed.
    """

    market_input = 1
    done = False
    step = 0
    print("Test 1:")
    while not done:
        step += 1
        llm_output, agent_action, state, reward, done, next_input, log_prob = env.step(market_input)
        print(f"Steg {step}: Input={market_input}, LLM={llm_output}, Action={agent_action}, Matches={state}")
        market_input = next_input  
    print("Reward:", reward)
