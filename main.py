from dataclasses import dataclass
import numpy as np
import torch
# import torchviz
from env import gym_env
from model import Network
from multiprocessing import Pipe, Process
import collections
import random
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TMP = False

act_repeat_time = 1
trajectory_size = 5
frame_siz = 4
Worker_num = 8
alpha = 0.5
beta = 0.01
gamma = 0.99
pic_width = 84
pic_height = 84
convert_shape = (-1, frame_siz, pic_width, pic_height)

@dataclass
class step_info:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: bool


def worker_func(worker_id, pipe):
    env = gym_env(worker_id)
    pict_queue = collections.deque(maxlen=frame_siz)
    _state = env.reset()
    for _ in range(frame_siz):
        pict_queue.append(_state)
    states = np.stack(pict_queue, axis=0).reshape(convert_shape)
    score = 0
    while True:
        cmd, act = pipe.recv()
        if cmd == "step":
            n_state, rew, done, info = env.step(act)
            pict_queue.append(n_state)
            score += rew
            n_state = np.stack(pict_queue, axis=0).reshape(convert_shape)
            ans = step_info(states, act, rew, n_state, done, info)
            states = n_state
            if info and (not done):
                # ans = step_info(states, act, -1.0, n_state, done, info)  # new
                tmp = random.randint(frame_siz, 10)
                pict_queue.clear()
                for _ in range(tmp):
                    n_state, _, _, _ = env.step(1)
                    pict_queue.append(n_state)
                states = np.stack(pict_queue, axis=0).reshape(convert_shape)
            pipe.send(ans)

        elif cmd == "reset":
            n_state = env.reset()
            pict_queue.clear()
            tmp = random.randint(frame_siz, 10)
            for _ in range(tmp):
                n_state, _, _, _ = env.step(1)
                pict_queue.append(n_state)
            states = np.stack(pict_queue, axis=0).reshape(convert_shape)
            score = 0
            pipe.send(states)

        elif cmd == "score":
            pipe.send(score)

        elif cmd == "test":
            print("id:{} : connect.".format(worker_id))

        elif cmd == "get_state":
            pipe.send(states)

        else:
            print("! error in worker func. id:{}".format(worker_id))


class Workers:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        self.self_pips = [pipe[0] for pipe in pipes]
        self.worker_pipes = [pipe[1] for pipe in pipes]

        self.workers = [Process(target=worker_func, args=(i, pipe)) for i, pipe in enumerate(self.worker_pipes)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        for pipe in self.self_pips:
            pipe.send(("test", None))

    def step(self, worker_id, action):
        self.self_pips[worker_id].send(("step", action))
        step = self.self_pips[worker_id].recv()
        return step

    def reset(self, worker_id):
        self.self_pips[worker_id].send(("reset", None))
        return self.self_pips[worker_id].recv()

    def score(self, worker_id):
        self.self_pips[worker_id].send(("score", None))
        return self.self_pips[worker_id].recv()

    def get_state(self, worker_id):
        self.self_pips[worker_id].send(("get_state", None))
        return self.self_pips[worker_id].recv()

    def one_step(self, acts, w_stops, w_dones):
        states = [self.get_state(i) for i in range(self.num_workers)]
        rews = [0 for _ in range(self.num_workers)]
        n_states = [self.get_state(i) for i in range(self.num_workers)]
        flags = [False for _ in range(self.num_workers)]

        for i in range(self.num_workers):
            if w_stops[i]:
                continue
            step = self.step(i, acts[i])
            rews[i] = step.reward
            n_states[i] = step.next_state
            w_dones[i] = step.done
            flags[i] = step.info
            if step.done:
                rews[i] = -1.0
            if step.info:
                rews[i] = -1.0

        return states, acts, rews, n_states, w_dones, flags

    def n_step(self, worker_id, act, prob, step_repeat_time):
        states = [self.get_state(worker_id)]
        rews = []
        n_states = []
        acts = []
        probs = []
        flag = False
        for i in range(step_repeat_time):
            step = self.step(worker_id, act)
            rews.append(step.reward)
            n_states.append(step.next_state)
            acts.append(act)
            probs.append(prob)
            if step.done:
                rews[-1] = -1.0  # new
                flag = True
                break

            if step.info:
                rews[-1] = -1.0  # new
                break

            if i != step_repeat_time - 1:
                states.append(step.next_state)

        return states, acts, rews, n_states, probs, flag




class Agent:
    def __init__(self):
        self.network = Network(action_space=4, step_repeat_times=frame_siz, alpha=alpha, beta=beta)
        self.workers = Workers(num_workers=Worker_num)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.global_score = collections.deque(maxlen=50)
        self.w_states = [self.workers.get_state(i) for i in range(self.workers.num_workers)]  # states of each workers.
        self.w_stop = []  # if true then do not worker.step()
        self.train_time = 0

    def calc_drew(self, rews, last_state):  # last_state は時系列的に最後のstate (n_states[-1]でok)
        d_rews = []
        ac_rew, _ = self.network(last_state)
        ac_rew = ac_rew[0].detach()
        for rew in rews:
            ac_rew = gamma * ac_rew + rew
            d_rews.append(ac_rew)
        d_rews.reverse()
        return d_rews

    def init_state(self, worker_id):
        score = self.workers.score(worker_id)
        self.global_score.append(score)
        print("ID[{}] : train time is {} : score is {}. average is {}.".format(worker_id, self.train_time, score, float(sum(self.global_score)) / float(
            len(self.global_score))))

        self.train_time += 1
        self.w_states[worker_id] = self.workers.reset(worker_id)

    def play_n_step2(self):
        _states = [[] for _ in range(self.workers.num_workers)]
        d_rews = []
        _acts = [[] for _ in range(self.workers.num_workers)]
        _probs = [[] for _ in range(self.workers.num_workers)]
        # for _ in range(trajectory_size):
        #     for i in range(self.workers.num_workers):
        #         act, prob = self.network.select_action(self.workers.get_state(i))
        #         #  print("act pns {}".format(act))
        #         steps = self.workers.n_step(i, act[0], prob[0], act_repeat_time)
        #         states.extend(steps[0])
        #         acts.extend(steps[1])
        #         d_rews.extend(self.calc_drew(steps[2], steps[3][-1]))
        #         probs.extend(steps[4])
        #         if steps[5]:
        #             self.init_state(i)
        #             break
        w_stops = [False for _ in range(self.workers.num_workers)]
        w_dones = [False for _ in range(self.workers.num_workers)]
        rews = [[] for _ in range(self.workers.num_workers)]
        last_state = [self.workers.get_state(i) for i in range(self.workers.num_workers)]
        for _ in range(trajectory_size):
            tmp = [self.network.select_action(lst) for lst in last_state]
            actions = [tp[0][0] for tp in tmp]
            probs = [tp[1][0] for tp in tmp]
            states2, acts2, rews2, n_states2, w_dones, flags = self.workers.one_step(actions, w_stops, w_dones)
            for j in range(self.workers.num_workers):
                if not w_stops[j]:
                    last_state[j] = n_states2[j]
                    _states[j].append(states2[j])
                    _acts[j].append(acts2[j])
                    _probs[j].append(probs[j])
                    rews[j].append(rews2[j])
                w_stops[j] = w_stops[j] or w_dones[j] or flags[j]

        states = []
        acts = []
        probs = []
        for i in range(self.workers.num_workers):
            d_rews.extend(self.calc_drew(rews[i], last_state[i]))
            states.extend(_states[i])
            acts.extend(_acts[i])
            probs.extend(_probs[i])

        for i, done in enumerate(w_dones):
            if done:
                self.init_state(i)

        return states, d_rews, acts, probs

    def play_n_step(self):
        states = []
        d_rews = []
        acts = []
        probs = []
        for _ in range(trajectory_size):
            for i in range(self.workers.num_workers):
                act, prob = self.network.select_action(self.workers.get_state(i))
                #  print("act pns {}".format(act))
                steps = self.workers.n_step(i, act[0], prob[0], act_repeat_time)
                states.extend(steps[0])
                acts.extend(steps[1])
                d_rews.extend(self.calc_drew(steps[2], steps[3][-1]))
                probs.extend(steps[4])
                if steps[5]:
                    self.init_state(i)
                    break

        return states, d_rews, acts, probs

    def train(self):
        self.network.to(dev)
        while True:
            states, d_rews, acts, probs = self.play_n_step()
            loss = self.network.calc_loss(states, acts, d_rews, probs)
            # dot = torchviz.make_dot(loss, params=dict(self.network.named_parameters()))
            # dot.format = 'png'
            # global TMP
            # if not TMP:
            #     dot.render('/home/emile/Documents/Code/breakout_A2C/graph_image')
            #     TMP = True
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()


def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()