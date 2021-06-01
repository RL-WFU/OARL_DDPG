OARL README
Joe McCalmon
5/25/2021

This repository adapts the work of Zhang et al. (2020) [1] for a defense mechanism against large perturbations,
Observation Agnostic Reinforcement Learning. Environments, base ddpg, sa-ddpg, and most of the code is from [1].
Models have been pre-trained. OARL runs on top of pre-trained models so the only instruction given here is how to
test ddpg, sa-ddpg, and oarl models given the pre-trained weights.

For installation of packages, refer to the original README.md file. The file does not mention that you must create
a student account to run Mujoco environments. In addition, I have slightly altered the requirements.txt file since
packages there had compatibility errors.

To Run Vanilla DDPG on the Inverted Pendulum Environment. No Attack:
eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg

With Attack:
eval_ddpg.py --config config/InvertedPendulum_vanilla.json --path_prefix models/vanilla-ddpg test_config:attack_params:enabled=true

To Run



