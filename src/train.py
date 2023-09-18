import numpy as np
from unityagents import UnityEnvironment
from collections import deque
import wandb
from utils import set_seeds, plot_scores, save_scores
from ppo_agent import PPOContinuous
import torch


def train(env, brain_name, ppo, n_episodes=2000, batch_size=128):
    scores_window = deque(maxlen=100)
    scores = []
    timesteps = 0
    max_score = 30.0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(states.shape[0])
        while True:
            actions, log_probs = ppo.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards

            ppo.memorize(states, actions, rewards, next_states, dones, log_probs)
            states = next_states
            timesteps += 1

            if any(dones):
                break
        ppo.update(batch_size=batch_size)
        mean_score = np.mean(score)
        wandb.log({"Episode Mean Score": mean_score})
        scores.append(mean_score)
        print(f"Episode {i_episode} Score: {mean_score}")

        scores_window.append(mean_score)
        mean_score_window = np.mean(scores_window)
        wandb.log({"Episode Mean Score Window": mean_score_window})

        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )

        if np.mean(scores_window) >= max_score:
            print(
                f"The environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window)}"
            )
            torch.save(
                ppo.actor_critic.state_dict(), "results/weights/actor_critic_128.pth"
            )
            max_score = np.mean(scores_window)

    return scores


if __name__ == "__main__":
    wandb.init(project="ppo_continuous")
    # Ensure reproducibility
    set_seeds(42)
    env = UnityEnvironment(file_name="./Reacher.app", seed=42, worker_id=5)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    ppo = PPOContinuous(state_size=state_size, action_size=action_size, logger=wandb)

    scores = train(env, brain_name, ppo, n_episodes=10000, batch_size=128)

    plot_scores(scores, "results/scores_128.png")
    save_scores(scores, "results/scores_128.npy")

    env.close()
    wandb.finish()
