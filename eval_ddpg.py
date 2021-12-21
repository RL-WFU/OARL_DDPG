#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from deep_rl import *
import sys
from argparser import argparser
from config import load_config
from OARL_prediction.State_Prediction_Policy_Model import PredictionWithPolicy

# from robust_ddpg import RobustDDPGAgent, RobustDeterministicActorCriticNet
#import numpy as np
#import torch
import copy
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def states_statics(states, logger):
    # states in shape (episode, frame, state_dim)
    mean = np.mean(states, axis=0)
    s_max = np.max(states, axis=0)
    s_min = np.min(states, axis=0)
    std = np.std(states, axis=0)
    s_range = list(zip(s_min, s_max))
    mean_str = '"state_mean": ['
    std_str = '"state_std":  ['
    for i, (i_mean, i_max, i_min, i_std) in enumerate(zip(mean, s_max, s_min, std)):
        logger.info('var %2d mean %8.4f std %8.4f min %8.4f max %8.4f', i, i_mean, i_std, i_min, i_max)
        mean_str += "{:8.5f}, ".format(i_mean)
        std_str += "{:8.5f}, ".format(i_std)
    mean_str = mean_str[:-2] + "],"
    std_str = std_str[:-2] + "]"
    print(mean_str)
    print(std_str)

def run_evalutation(config, agent):
    all_states = []
    all_actions = []
    episodic_returns = []
    for ep in range(agent.config.eval_episodes):
        total_rewards, states, actions, certify_loss_l1, certify_loss_l2, certify_loss_linf, certify_loss_range, tran, mae = agent.eval_episode(show=agent.config.show_game, return_states=True,
                                                    certify_eps=agent.config.certify_params["eps"] if agent.config.certify_params["enabled"] else 0.0, episode_number=ep)

        if ep == 0:
            transitions = np.asarray(tran)
        else:
            transitions = np.concatenate([transitions, np.asarray(tran)], axis=0)

        
        agent.logger.info('epoch %d reward %f steps %d', ep, total_rewards, len(states))
        episodic_returns.append(np.sum(total_rewards))
        all_states.append(np.array(states))
        all_actions.append(np.array(actions))
    
    return all_states, all_actions, episodic_returns, transitions
    

def single_ddpg_eval(config):
    """
    This function handles evaluation of the agent under the attack parameters specified in the config file. General use-case function for evaluation.
    """
    agent = RobustDDPGAgent(config)
    agent.logger.info('Evaluation started!\n %s', str(config.__dict__))
    best_model = os.path.join(agent.config.models_path, "model_best") #ALTER BEST ATTACK MODEL
    agent.logger.info('Loading %s', best_model)
    # agent.load(best_model)
    if config.attack_params['type'].startswith('sarsa'):
        agent.load_sarsa(best_model)
    else:
        agent.load(best_model)

    if config.save_transition_path is not None:
        print("will save transition")
    else:
        print("will not save transition")

    all_states, all_actions, episodic_returns, transitions = run_evalutation(config, agent)

    # Each episode may have different length, so we concat all to one
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    mean_return = np.mean(episodic_returns)
    agent.logger.info('states variable statistics')
    states_statics(all_states, agent.logger)
    agent.logger.info('action variable statistics')
    states_statics(all_actions, agent.logger)
    agent.logger.info('Average Reward: %f, min reward: %f, std: %f, max: %f', mean_return, np.min(episodic_returns), np.std(episodic_returns), np.max(episodic_returns))
    agent.logger.info('[END-END-END] Finishing evaluation')

    if config.save_transition_path is not None:
        outfile = config.save_transition_path
        outfile = outfile + "_transition.npy"

        transitions = np.asarray(transitions)
        np.save(outfile, transitions)



def repeated_ddpg_eval(config):
    agent = RobustDDPGAgent(config)
    agent.logger.info('Evaluation started!\n %s', str(config.__dict__))
    best_model = os.path.join(agent.config.models_path, "model_best") #ALTER BEST ATTACK MODEL
    agent.logger.info('Loading %s', best_model)
    if config.attack_params['type'].startswith('sarsa'):
        agent.load_sarsa(best_model)
    else:
        agent.load(best_model)

    if config.save_transition_path is not None:
        print("will save transition")
    else:
        print("will not save transition")



    epsilons = [0, .3, 1.5, 5]
    rates = [1, 0.75, 0.5, 0.25, 0]
    """
    Repeated evaluation loops through different levels of epsilon and attack rate, evaluating the agent according to the procedure in run_evaluation. Used for quick testing and plotting of different attack parameters in one run.
    """
    for r in rates:
        agent.set_attack_rate(r)
        print("Attack rate: {}", r)

        for e in epsilons:

            agent.set_epsilon(e)
            print("Epsilon: {}", e)
    

            all_states, all_actions, episodic_returns, transitions = run_evalutation(config, agent)

            # Each episode may have different length, so we concat all to one
            all_states = np.concatenate(all_states, axis=0)
            all_actions = np.concatenate(all_actions, axis=0)
            mean_return = np.mean(episodic_returns)
            agent.logger.info('states variable statistics')
            states_statics(all_states, agent.logger)
            agent.logger.info('action variable statistics')
            states_statics(all_actions, agent.logger)
            agent.logger.info('Average Reward: %f, min reward: %f, std: %f, max: %f', mean_return, np.min(episodic_returns), np.std(episodic_returns), np.max(episodic_returns))
            agent.logger.info('[END-END-END] Finishing evaluation')

            if config.save_transition_path is not None:
                outfile = config.save_transition_path
                outfile = outfile + "_transition.npy"

                transitions = np.asarray(transitions)
                np.save(outfile, transitions)




# config_dict is our configurations set in JSON file
def ddpg_eval_setup(config_dict,suffix="", eval=True):
    # Read config file and translate relevant fields
    ddpg_config = {}
    ddpg_config['game'] = config_dict['env_id']
    eval_config = config_dict['test_config']
    train_config = config_dict['training_config']
    
    generate_tag(ddpg_config)
    ddpg_config.setdefault('log_level', 0)
    config = Config()
    config.merge(ddpg_config)
    config.training = False
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    print(config.eval_env)
    config.eval_episodes = eval_config['eval_episodes']
    config.tag += "_eval"

    config.OARL = eval_config['OARL']
    if config.OARL:
        config.OARL_params = eval_config['OARL_params']
        config.torch_oarl = PredictionWithPolicy
        print("OARL PARAMS: {}".format(config.OARL_params))


    if config_dict["test_config"]["save_transition_path"] is not "":
        config.save_transition_path = config_dict["test_config"]["save_transition_path"]
    else:
        config.save_transition_path = None
    # newly added properties
    config.models_path = config_dict["models_path"]
    config.show_game = eval_config['show_game']
    config.save_frame = eval_config['save_frame']
    config.actor_network = train_config["actor_network"]
    config.critic_network = train_config["critic_network"]
    config.robust_params = train_config["robust_params"]
    # Setting beta scheduler to match the parameters of epsilon scheuler
    if config.robust_params['beta_scheduler']['start_step'] == 'auto':
        config.robust_params['beta_scheduler']['start_step'] = config.robust_params['eps_scheduler']['start_step']
        print('robust_params[beta_scheduler][start_step] is setting to', config.robust_params['beta_scheduler']['start_step'])
    if config.robust_params['beta_scheduler']['steps'] == 'auto':
        config.robust_params['beta_scheduler']['steps'] = config.robust_params['eps_scheduler']['steps']
        print('robust_params[beta_scheduler][steps] is setting to', config.robust_params['beta_scheduler']['steps'])
    # Setting adv training scheduler eps to match the parameters of epsilon scheuler
    if config.robust_params['advtrain_scheduler']['start'] == 'auto':
        config.robust_params['advtrain_scheduler']['start'] = config.robust_params['eps_scheduler']['end']
        print('robust_params[advtrain_scheduler][start] is setting to', config.robust_params['advtrain_scheduler']['start'])
    if config.robust_params['advtrain_scheduler']['end'] == 'auto':
        config.robust_params['advtrain_scheduler']['end'] = config.robust_params['eps_scheduler']['end']
        print('robust_params[advtrain_scheduler][end] is setting to', config.robust_params['advtrain_scheduler']['end'])
    config.certify_params = eval_config['certify_params']
    config.attack_params = eval_config["attack_params"]
    #This is perturbation per step
    print("ATTACK EPSILON IS SET TO", config.attack_params['eps'])
    if config.attack_params['alpha'] == "auto":
        config.attack_params['alpha'] = config.attack_params['eps'] / config.attack_params['iteration'] #Adjust config.attack_params['eps'] to adjust attack eps
        print('config.attack_params[alpha] is setting to', config.attack_params['alpha'])
    if config.certify_params['eps'] == "auto":
        config.certify_params['eps'] = config.attack_params['eps']
        print('config.certify_params[eps] is setting to', config.certify_params['eps'])
    config.data_params = config_dict["data_config"]
    config.sarsa_params = eval_config["sarsa_params"]
    config.noise_sigma = float(eval_config['noise_sigma'] )

    global RobustDDPGAgent, RobustDDPGAgent
    if config.attack_params['enabled'] == True and config.attack_params['type'].startswith('sarsa'):
        # from sarsa_ddpg import RobustDDPGAgent, RobustDeterministicActorCriticNet
        config.tag += "_" + suffix
    # else:
    #     from robust_ddpg import RobustDDPGAgent, RobustDeterministicActorCriticNet

    print("NETWORK FUNCTION PARAMS: ", config.state_dim, config.action_dim, config.actor_network, config.critic_network, config.mini_batch_size, config.certify_params)

    config.network_fn = lambda: RobustDeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        config.actor_network, config.critic_network,
        config.mini_batch_size,
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=0.0),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=0.0),
        robust_params=config.certify_params)

    config.replay_fn = lambda: None
    config.random_process_fn = lambda: None

    sys.stdout.flush()
    sys.stderr.flush()

    if config['repeat_test']:
        repeated_ddpg_eval(config)
    else:
        single_ddpg_eval(config)

USE_CUDA = torch.cuda.is_available()
def main(args):
    config = load_config(args)
    print(config["test_config"]["save_transition_path"])
    # game_list = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'Ant-v2', 'Humanoid-v2']
    config['models_path'] = os.path.join(args.path_prefix, config['models_path'].format(env_id=config['env_id']))
    mkdir(config['models_path'])
    mkdir(os.path.join(config['models_path'], 'checkpoints'))
    mkdir(os.path.join(config['models_path'], 'log'))
    mkdir(os.path.join(config['models_path'], 'tf_log'))
    mkdir(os.path.join(config['models_path'], 'runtime'))

    if args.repeat_test is not None:
        config['repeat_test'] = args.repeat_test
    else:
        config['repeat_test'] = False


    print("Use_cuda: {}".format(USE_CUDA))
    if not USE_CUDA:
    #if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
        # Use CPU.
        #del os.environ['CUDA_VISIBLE_DEVICES'] # Avoid HIP error.
        #if 'HIP_VISIBLE_DEVICES' in os.environ:
            #del os.environ['HIP_VISIBLE_DEVICES'] # Avoid HIP error.
        print('Using CPU.')
        select_device(-1)
    else:
        print('Using GPU.')
        select_device(0)
    suffix = ""
    if config['test_config']['attack_params']['enabled'] == True and config['test_config']['attack_params']['type'].startswith('sarsa'):

        # if need to train a model or not
        scheduler_opts = config['test_config']['sarsa_params']['action_eps_scheduler']
        suffix = "_{start}_{end}_{steps}_{start_step}".format(**scheduler_opts) + "_{}".format(config['test_config']['sarsa_params']['sarsa_reg'])
        sarsa_model_filename = os.path.join(config['models_path'], "model_best.model_sarsa") + suffix

        if not os.path.exists(sarsa_model_filename):
            # need to train a model 
            print(f"Existing value function {sarsa_model_filename} does not exist. Will train a new one.")
            from train_sarsa import ddpg_continuous
            #ddpg_eval_setup(config)
            ddpg_continuous(config)
    global RobustDDPGAgent, RobustDeterministicActorCriticNet
    # evaluation 
    from robust_ddpg import RobustDDPGAgent, RobustDeterministicActorCriticNet
    

    ddpg_eval_setup(config, suffix=suffix)

if __name__ == "__main__":
    args = argparser()
    main(args)

