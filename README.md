# Observation Agnostic Reinforcement Learning

This repository adapts the work of [1] for a defense mechanism against large perturbations,
Observation Agnostic Reinforcement Learning. Environments, base ddpg, sa-ddpg, and most of the code is from [1].
Models have been pre-trained. OARL runs on top of pre-trained models so the only instruction given here is how to
test ddpg, sa-ddpg, and oarl models given the pre-trained weights. For instructions on how to pre-train models using this repository, see saddpg_readme.md

For installation of packages, refer to saddpg_readme.md file. This installation requires Mujoco version 1.50, which requires a (free) license.

## Usage

To Run Vanilla DDPG on the Inverted Pendulum Environment. 

### No Attack:

eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg

### With Attack:

eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true

### For Repeated Evaluation with Different Attack Parameters (specified inside eval_ddpg.py)

eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg --repeat_test True test_config:attack_params:enabled=true 

### To run OARL for Inverted Pendulum

eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true  test_config:OARL=true

### To run Inverted Pendulum vanilla and save transitions

eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg test_config:save_transition_path="transitions/pend_transitions"

### General Usage:

General configuration parameters are set in the **defaults.json** file. To run evaluation, we specify an additional config file, i.e **InvertedPendulum_vanilla.json** which overrides some parameters in defaults.json.
The specific config file designates the environment, attack epsilon, and parameters for the chosen type of trained model. If a path prefix of models/vanilla-ddpg is used, then {env}_vanilla.json is the config file which should be used.
{env}_robust.json corresponds to models trained with saddpg, while {env}_adv_train.json is for models trained with adversarial training.

To enable/disable the OARL defense mechanism see defaults.json. Ensure test_config:OARL=true
Specific OARL detection model paths and other OARL parameters should be controlled via the specific config file, or through command line arguments.

Implementation of attacks are done in robust_ddpg.py
Implementation of OARL agent is in deep_rl/agent/BaseAgent.py

## Reference

[1]*Huan Zhang\*, Hongge Chen\*, Chaowei Xiao, Bo Li, Duane Boning,* and *Cho-Jui
Hsieh*, "**Robust Deep Reinforcement Learning against Adversarial Perturbations
on State Observations**". [**NeurIPS 2020
(Spotlight)**](https://proceedings.neurips.cc/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf)
(\*Equal contribution)

