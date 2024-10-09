
from pettingzoo.classic import texas_holdem_v4


def decide_action(environment):
    #always fold
    return 2

def main():
    env = texas_holdem_v4.env(num_players=4, render_mode="human", screen_height = 500)
    env.reset(seed=42)
    env.render()
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            #action = env.action_space(agent).sample(mask)
            action = decide_action(env)
            print(action)

        env.step(action)
    env.close()


if __name__=="__main__":
    main()