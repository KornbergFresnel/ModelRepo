import gym
from model import DQN
from config import GeneralConfig


def test(env, step, agent):
    obs = env.reset()
    total_reward = 0
    total_step = 0

    for _ in range(step):
        # env.render()
        action = agent.pick_action(obs, train=False)
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        total_step += 1

        if done:
            break

    agent.reward_record.append(total_reward)

    mess = "[* TEST] Reward: {0:.3f}, TotalStep: {1}".format(total_reward, total_step)

    print(mess)


def main(_config):
    env = gym.make(_config.ENV_NAME)

    agent = DQN(env, _config)

    print("[*] --- Begin Emulator Training ---")

    for episode in range(_config.EPISODE):

        obs = env.reset()

        # === Emulator ===
        for i in range(_config.STEP):
            action = agent.pick_action(obs)
            obs_next, reward, done, _ = env.step(action)

            # agent will store the newest experience into replay buffer, and training with mini-batch and off-policy
            agent.perceive(obs, action, reward, done)

            if done:
                break

            obs = obs_next

        # == train ==
        agent.train(episode)

        if (episode + 1) % agent.save_every == 0:
            agent.save(step=episode)

        # == test ==
        print("\n[*] === Enter TEST module ===")
        test(env, _config.STEP, agent)

    agent.record()


if __name__ == "__main__":
    config = GeneralConfig()
    main(config)
