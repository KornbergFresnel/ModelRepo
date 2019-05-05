import os
import sys
import argparse
import operator
import tensorflow as tf
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../lib/ma_env'))

from lib import multiagent
from MADDPG.model import MultiAgent


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'log')
MODEL_BACK_UP = os.path.join(BASE_DIR, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--name', type=str, default='maddpg', help='Naming for logging.')
    parser.add_argument('--scenario', type=str, default='simple_push.py',
                        help='Path of the scenario Python script (default=push_ball.py).')

    parser.add_argument('--n_agent', type=int, default=2, help='Set the number of agents (default=2')
    parser.add_argument('--len_episode', type=int, default=25, help='Time horizon limitation (default=25).')
    parser.add_argument('--n_train', type=int, default=100, help='Training round.')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation episode interval (default=50).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default=64).')
    parser.add_argument('--memory_size', type=int, default=10**5, help='Memory size (default=10**5).')
    parser.add_argument('--load', type=int, default=0, help='Load existed model.')

    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Setting learning rate for Actor (default=1e-4).')
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='Setting learning rate for Critic (default=1e-3).')
    parser.add_argument('--tau', type=float, default=0.01, help='Hyper-parameter for soft update (default=0.01).')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor (default=0.98).')

    parser.add_argument('--render', action='store_true', help='Turn on render or not.')
    args = parser.parse_args()

    # =========================== initialize environment =========================== #
    step, steps_limit = 0, args.len_episode * args.n_train
    scenario = multiagent.scenarios.load(args.scenario).Scenario()
    world = scenario.make_world(num_agents=args.n_agent, world_dim_c=1, num_landmarks=1, num_adversaries=args.n_agent // 2)

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                               info_callback=None, shared_viewer=True)

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    maddpg = MultiAgent(sess, env, args.name, args.n_agent, args.batch_size, args.actor_lr, args.critic_lr, args.gamma,
                        args.tau, args.memory_size)

    # initialize summary
    summary_r = [None for _ in range(env.n)]

    for i in range(env.n):
        summary_r[i] = tf.placeholder(tf.float32, None)
        tf.summary.scalar('Episode-Reward-{}'.format(i), summary_r[i])

    summary_dict = {'reward': summary_r}

    if not args.render:
        summary_a_loss, summary_c_loss = [None for _ in range(env.n)], [None for _ in range(env.n)]
        for i in range(env.n):
            summary_a_loss[i] = tf.placeholder(tf.float32, None)
            summary_c_loss[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar('Actor-Loss-{}'.format(i), summary_a_loss[i])
            tf.summary.scalar('Critic-Loss-{}'.format(i), summary_c_loss[i])

        summary_dict['a_loss'] = summary_a_loss
        summary_dict['c_loss'] = summary_c_loss

    merged = tf.summary.merge_all()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    maddpg.init()  # run self.sess.run(tf.global_variables_initializer()) and hard update

    if args.load > 0:
        maddpg.load(os.path.join(MODEL_BACK_UP, args.name), epoch=args.load)

    # ======================================== main loop ======================================== #
    a_loss, c_loss = None, None
    if args.render:
        env.render()
    else:
        a_loss = [[] for _ in range(env.n)]
        c_loss = [[] for _ in range(env.n)]

    obs_n = env.reset()
    episode_r_n = [0. for _ in range(env.n)]

    # update this flag every `len_episode * eval_interval` steps, if it is true, then no training and data collection
    is_evaluate = False

    while step < steps_limit:
        act_n = maddpg.act(obs_n)
        next_obs_n, reward_n, done_n, info_n = env.step(act_n)

        if not args.render and not is_evaluate:  # trigger for data collection
            maddpg.store_trans(obs_n, act_n, next_obs_n, reward_n, done_n)

        obs_n = next_obs_n
        episode_r_n = map(operator.add, episode_r_n, reward_n)

        step += 1

        # =============================== render / record / model saving ===============================
        if args.render:
            env.render()
        else:
            if not is_evaluate:  # trigger for training
                t_info_n = maddpg.train()

                if t_info_n is not None:
                    a_loss = map(lambda x, y: y.append(x), t_info_n['a_loss'], a_loss)
                    c_loss = map(lambda x, y: y.append(x), t_info_n['c_loss'], c_loss)

        if step % args.len_episode == 0 or np.any(done_n):
            obs_n = env.reset()

            feed_dict = dict()

            if args.render or is_evaluate:
                feed_dict.update(zip(summary_dict['reward'], episode_r_n))
                episode_r_n = [0. for _ in range(env.n)]

            if not args.render and is_evaluate:
                a_loss = map(lambda x: sum(x) / len(x), a_loss)
                c_loss = map(lambda x: sum(x) / len(x), c_loss)

                feed_dict.update(zip(summary_dict['a_loss'], a_loss))
                feed_dict.update(zip(summary_dict['c_loss'], c_loss))

                a_loss = [[] for _ in range(env.n)]
                c_loss = [[] for _ in range(env.n)]

                maddpg.save(MODEL_BACK_UP, step // args.len_episode)

            if args.render or is_evaluate:
                summary = sess.run(merged, feed_dict=feed_dict)
                summary_writer.add_summary(summary, (step - 1) // args.len_episode)

            is_evaluate = (step // args.len_episode % args.eval_interval == 0)
