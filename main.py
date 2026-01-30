import torch
from agent import Agent
from environment import EnvWF
from config import Config


if __name__ == '__main__':

    env = EnvWF(snr_norm=Config.SNR_NORM_FACTOR, angle_norm=Config.ANGLE_NORM_FACTOR)
    agent_params = Config.TRAIN_PARAMS.copy()
    agent_params['env'] = env
    agent_params['net_config'] = Config.AGENT_INIT

    agent = Agent(**agent_params)

    eval_every = 500
    cum_reward = 0
    cum_loss = 0
    cum_constraint = 0

    average_reward_list = []
    average_loss_list = []
    average_constraint_list = []

    for episode in range(60000):
        s = env.ge_rnd_state(episode)
        print(s)
        for u in range(3):
            a, mc = agent.act(s[u], episode)
            effi = env.step(s[u], a, mc)
            agent.put(s[u], a, mc, effi)
            loss = agent.learn(episode)

    # save the networks
    torch.save(agent.actor, "./results/model/actor.pt")
    torch.save(agent.Qnet, "./results/model/Qnet.pt")
    torch.save(agent.multiplier, "./results/model/multiplier.pt")









