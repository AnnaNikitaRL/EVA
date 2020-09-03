# Ephemeral Value Adjustment

This repository contains an implementation of Ephemeral Value Adjustment (EVA) from "Fast deep reinforcement learning using online adjustments from the past" by S.Hansen et al. https://arxiv.org/abs/1810.08163 </br>
Project is done as part of the course <a href="http://deeppavlov.ai">DeepPavlov </a> course: <a href="http://deeppavlov.ai/rl_course_2020"> Advanced Topics in Deep Reinforcement learning </a>. 

#### <a href="https://github.com/amfolity/">Anna Mazur</a>, <a href="https://github.com/darthrevenge">Nikita Trukhanov</a>

### Prerequisities
We used PyTorch library version 1.6.</br>
For performing approximate nearest neighbours search we used Fast Library for Approximate Nearest Neighbors (FLANN).
Original codes and install instructions could be found https://github.com/mariusmuja/flann.

### Running the code

In order to run the code with the default parameters use the snippet below. The deafult parameters can be found in the module config.py in our repository.

```sh
python experiment.py
```

In order to run baseline DQN model one can switch off weighting parameter &lambda;

```sh
python experiment.py --lambd=0
```

### Parameters and some differences from the original article
We ran experiments primarly on atari environments, such as "BreakoutNoFrameskip-v4" and "AtlantisNoFrameskip-v4". We used EpisodicLifeEnv, FireReset or NoOpReset and MaxAndSkipEnv wrappers from OpenAI.baselines. We did not use four parallel agents like in original work. We also limit the size of experience replay to 400k. Other than that, we tweeked a little bit some parameters: greedy exploration rate (&varepsilon;) decay rate, Adam learning rate. Most of the parameters were kept the same as in the original work (Section 9 Atari Experiment Details).</br>

### Results
As a benchmark we have used DQN of exactly the same atchitecture and hyperparameters. The only difference is that in DQN, weighting parameter for non-parametric Q-value, &lambda; equals 0. </br>
Also, we implemented 
 
