import sys

import numpy as np
from dataset import *

import argparse

sys.path.append('.')
from domains.gridworld import *
from generators.obstacle_gen import *
sys.path.remove('.')


def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of
    #  actions corresponding to the trajectory.
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    # action_vecs[4:]表示第四行到最后一行
    # 动作是斜向的，全部除以根号2
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    # 对所有元素进行转置
    action_vecs = action_vecs.T
    # 轨迹是个(1,n,2)维的tensor，第一维表示一条轨迹，第二维和第三维表示坐标位置
    # axis=0就是在同一条轨迹上求坐标的差，x互减，y互减
    # 求完差以后，输出是一个(1,n-1,2)维的tensor
    state_diff = np.diff(traj, axis=0)
    # 给坐标位置差求平方
    state_diff_square = np.square(state_diff)
    # print("state_diff_square: \n", state_diff_square)
    # axis=1 表示把第二维度的每一行加起来
    # 比如 [[1 1] [1 1] [0 1]]，按第二维度加起来得到[2 2 1]，tensor降低了一个维度
    state_diff_square_sum = np.sum(state_diff_square, axis=1)
    state_diff_square_sum_sqrt = np.sqrt(state_diff_square_sum)
    state_diff_square_sum_sqrt_reciprocal = 1 / state_diff_square_sum_sqrt
    # 扩展为2行1列
    state_diff_square_sum_sqrt_reciprocal_tile = \
        np.tile(state_diff_square_sum_sqrt_reciprocal, (2, 1))
    state_diff_square_sum_sqrt_reciprocal_tile_transpose = state_diff_square_sum_sqrt_reciprocal_tile.T
    # 从求差到乘以差的过程，是在计算delta_x，delta_y围成的直角三角形的三角函数值
    # 求这个有何用？
    norm_state_diff = state_diff * state_diff_square_sum_sqrt_reciprocal_tile_transpose
    # print("norm_state_diff: \n", norm_state_diff)
    # print("action_vecs: \n", action_vecs)
    # 点积，前后都是多维的，按照矩阵乘法计算
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    # print("prj_state_diff: \n", prj_state_diff)
    # print("-----: \n", prj_state_diff - 1)
    # actions_one_hot每行只会有一个数为真，表示了这次移动的方向
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    # print("actions_one_hot: \n", actions_one_hot)
    # print("np.arange(n_actions).T: \n", np.arange(n_actions).T)
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    # print("actions: \n", actions)
    # 返回了每条轨迹中从起点到终点，每走一个格子的动作是action_vecs的第几个
    return actions

# state_batch_size在生成数据的时候未被使用
def make_data(dom_size, n_domains, max_obs, max_obs_size, n_traj,
              state_batch_size):

    X_l = []
    S1_l = []
    S2_l = []
    Labels_l = []

    dom = 0.0
    # 一个domains中有一张地图，一个目的地点，几个障碍物，多个起始点
    # 一个domains，一个trajectory中有一张地图，一个目的地点，几个障碍物，一个起点
    # 每生成1个栅格地图，在里边指定设定的多条轨迹；然后生成另1个栅格地图
    while dom <= n_domains:
        # 随机生成一个目标点，坐标是整型的x，y
        goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
        # Generate obstacle map
        obs = obstacles([dom_size[0], dom_size[1]], goal, max_obs_size)
        # Add obstacles to map
        # 在障碍物的个数限值的范围内实际生成的障碍物个数
        n_obs = obs.add_n_rand_obs(max_obs)
        # Add border to map
        border_res = obs.add_border()
        # Ensure we have valid map
        if n_obs == 0 or not border_res:
            continue
        # Get final map
        # 得到障碍物地图
        im = obs.get_final()
        # obs.show()
        # Generate gridworld from obstacle map
        # 得到最终地图，带有目的地点
        G = GridWorld(im, goal[0], goal[1])
        # Get value prior
        # 目的地点的值为10，其他为0
        value_prior = G.t_get_reward_prior()
        # Sample random trajectories to our goal
        # states_one_hot中有路径的元素为1，没有路径的元素为0
        states_xy, states_one_hot = sample_trajectory(G, n_traj)
        for i in range(n_traj):
            # 一条轨迹有一组xy坐标
            if len(states_xy[i]) > 1:
                # Get optimal actions for each state
                # 就是把采样轨迹的每一步的动作拿出来
                actions = extract_action(states_xy[i])
                # 表示某条轨迹从起点到终点走了几步
                ns = states_xy[i].shape[0] - 1
                # print("ns: ", ns)
                # Invert domain image => 0 = free, 1 = obstacle
                image = 1 - im
                # print("image: \n", image)
                # Resize domain and goal images and concate
                image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
                value_data = np.resize(value_prior,
                                       (1, 1, dom_size[0], dom_size[1]))
                # image_data中0表示可行无障碍，1表示有障碍
                # value_data中10表示目的地点，其他为0
                # image_data是4维，第一维表示第几个域，第二维表示第几条轨迹，第三维表示行，第四维表示列
                # concatenate拼接， axis=1， 表示同一条轨迹对应的地图和目标点奖赏地图拼接， 拼接为2层
                iv_mixed = np.concatenate((image_data, value_data), axis=1)
                # ns表示某条轨迹从起点到终点走了几步，每走一步，就在第一维拼上1层
                X_current = np.tile(iv_mixed, (ns, 1, 1, 1))
                # Resize states
                # S1,S2组成轨迹，但是不包含最后一个目的地点
                S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1)
                S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1)
                # Resize labels
                # 保存的是动作的序列
                Labels_current = np.expand_dims(actions, axis=1)
                # Append to output list
                X_l.append(X_current)
                S1_l.append(S1_current)
                S2_l.append(S2_current)
                Labels_l.append(Labels_current)
        dom += 1
        sys.stdout.write("\r" + str(int((dom / n_domains) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Concat all outputs
    X_f = np.concatenate(X_l)
    S1_f = np.concatenate(S1_l)
    S2_f = np.concatenate(S2_l)
    Labels_f = np.concatenate(Labels_l)
    print("X_f.shape: ", X_f.shape)
    return X_f, S1_f, S2_f, Labels_f


def main(dom_size=(28, 28),
         n_domains=5000,
         max_obs=50,
         max_obs_size=2,
         n_traj=7,
         state_batch_size=1):
    # Get path to save dataset
    # save_path = "dataset/gridworld_{0}x{1}".format(dom_size[0], dom_size[1])
    save_path = "gridworld_{0}x{1}".format(dom_size[0], dom_size[1])
    # Get training data
    print("Now making training data...")
    X_out_tr, S1_out_tr, S2_out_tr, Labels_out_tr = make_data(
        dom_size, n_domains, max_obs, max_obs_size, n_traj, state_batch_size)
    # Get testing data
    print("\nNow making  testing data...")
    X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts = make_data(
        dom_size, n_domains / 6, max_obs, max_obs_size, n_traj,
        state_batch_size)
    # Save dataset
    # 把生成的训练数据和测试数据保存在同一个.npz文件中
    np.savez_compressed(save_path, X_out_tr, S1_out_tr, S2_out_tr,
                        Labels_out_tr, X_out_ts, S1_out_ts, S2_out_ts,
                        Labels_out_ts)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # default 28
    parser.add_argument("--size", "-s", type=int, help="size of the domain", default=8)
    # default 5000
    parser.add_argument("--n_domains", "-nd", type=int, help="number of domains", default=7)
    # default 50
    parser.add_argument("--max_obs", "-no", type=int, help="maximum number of obstacles", default=3)
    parser.add_argument("--max_obs_size", "-os", type=int, help="maximum obstacle size", default=3)
    # default 7
    parser.add_argument("--n_traj", "-nt", type=int, help="number of trajectories", default=4)
    parser.add_argument("--state_batch_size", "-bs", type=int, help="state batch size", default=1)

    args = parser.parse_args()
    size = args.size

    main(dom_size=(size, size), n_domains=args.n_domains, max_obs=args.max_obs,
         max_obs_size=args.max_obs_size, n_traj=args.n_traj, state_batch_size=args.state_batch_size)
