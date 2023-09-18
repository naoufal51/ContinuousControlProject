## PPO Continuous Training for Unity Environments

### Description

This project revolves around training DRL agents to solve Unity environments that present continuous action spaces. We choose to adopt Proximal Policy Optimization (PPO) algorithm because of its efficiency and stability during training compared to other policy gradient methods, the implementation aims to solve the "Reacher" environment. The repository provides the necessary code for training and evaluation. We also addded the artifacts generated during training (weights).

https://github.com/naoufal51/ContinuousControlProject/assets/15954923/129aa305-1186-4d3e-9a03-c2fdd8c132cb


<!-- !["Reacher"](images/reacher.gif) -->
### The Environment - Unity's Reacher

In this Unity Reacher environment, agents can control a double-jointed arm to reach predefined target locations. The agent needs to maintain the arm's position within the target location as long as possible. The environment provides a reward of +0.1 for every time the agent's hand/last joint is within the target location.

**Observation and Action Spaces**:
- **Observation Space**: To sense the environment, the agent relies on observing a 33-variable state. The state incorporates the position, rotation, velocity, and angular velocities of the arm.
- **Action Space**: The agent can control the arm by specifying the torques applicable to the two joints. Therefore, generating a four-dimensional vector representing the action. These values are limited to a range between -1 and 1.

**Distributed Training Variants**:
1. **Single Agent Environment**: A single agent learning.
2. **Multiple Agent Environment**: 20 identical agents operating in independent environments. This version is tailored for algorithms like PPO and A3C which leverage parallelism.

**Solving the Environment**:
- **Option 1 - Single Agent**: The agent is considered proficient when achieving an average score of +30 across 100 successive episodes.
- **Option 2 - Multiple Agents**: The 20-agent environment is considered mastered once the collective average score reaches +30 over 100 consecutive episodes.

**Here we are dealing with the Single agent Scenario**

### Installation and Dependencies
#### Prerequisites

1. **Python Environment**: Ensure Python (>=3.6) is installed. Due to package dependencies, it's advised to set up the `drlnd` kernel. Comprehensive setup guide available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Activation command:
    ```bash
    source activate drlnd
    ```

2. **Weights & Biases Integration**: For real-time performance tracking, analysis, and hyperparameter tuning, Weights & Biases is integrated. Registration might be required for first-time users.

#### Installation
1. Clone the repository to access the latest features and implementations:
    ```bash
    git clone https://github.com/naoufal51/ContinuousControlProject.git
    ```

2. Transition to the project's root directory and install the pertinent dependencies:
    ```bash
    cd ContinuousControlProject
    pip3 install -r requirements.txt
    ```

3. Synchronize Weights & Biases with your unique API key to facilitate seamless experiment tracking:
    ```bash
    export WANDB_API_KEY=<your_wandb_api_key>
    ```

**Dependencies**:
- `numpy`: For numerical operations and data manipulation.
- `torch`: Deep learning framework used for modeling and optimization.
- `unityagents`: Interface and utilities for Unity-based environments.
- `wandb`: Performance tracking and visualization tool.
- `matplotlib`: For creating static, animated, and interactive visualizations.

### Usage
#### Training:
1. Ensure the Unity Environment executable (`Reacher.app`) resides in the project's root directory.
2. Command the training routine, sitting back as the agent learns:
   ```bash
   python3 src/train.py
   ```

#### Evaluation:
To observe the prowess of a trained agent, utilizing the stored weights, proceed with:
```bash
python3 src/evaluate.py
```

#### Visualizations:
Peruse the `results` directory for intricate plots elucidating the agent's training progression. Trained model parameters are archived in the `results/weights` directory.


### Credits
This project draws inspiration from :
1. Udacity DRLND.
2. [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)

### License
The content and implementations of this repository are shielded by the [MIT License](<link_to_license>).

