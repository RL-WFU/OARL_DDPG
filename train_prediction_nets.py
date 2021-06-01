from argparser import argparser
from config import load_config
from deep_rl.utils.config import Config
from deep_rl.agent import BaseAgent
import tensorflow as tf
import numpy as np
from OARL_prediction.State_Prediction_Policy_Model import train_pendulum

if __name__ == '__main__':

    args = argparser()
    config_dict = load_config(args)
    train_config = config_dict['training_config']
    config = Config()
    config.merge(config_dict)

    config.state_dim = train_config['state_dim']
    config.action_dim = train_config['action_dim']
    config.actor_network = train_config['actor_network']
    config.critic_network = train_config['critic_network']
    config.pol_reg = train_config['pol_reg']
    train_pendulum(config, config.pol_reg)
    print("NETWORK FUNCTION PARAMS: ", config.state_dim, config.action_dim, config.actor_network, config.critic_network,
          config.mini_batch_size, config.certify_params)



