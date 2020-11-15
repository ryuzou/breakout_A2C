from dataclasses import dataclass
import numpy as np
import torch
from gym import wrappers
from env import gym_env
from model import Network
from multiprocessing import Pipe, Process
import collections
import torchviz

TMP = False

act_repeat_time = 5
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
    env = gym_env()
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
            if info:
                n_state, _, _, _ = env.step(1)
                pict_queue.clear()
                for _ in range(frame_siz):
                    pict_queue.append(n_state)
                states = np.stack(pict_queue, axis=0).reshape(convert_shape)
            pipe.send(ans)

        elif cmd == "reset":
            n_state = env.reset()
            n_state, _, _, _ = env.step(1)
            pict_queue.clear()
            for _ in range(frame_siz):
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

    # def step(self, actions):
    #     for pipe, act in zip(self.self_pips, actions):
    #         pipe.send(("step", act))
    #
    #     steps = [pipe.recv() for pipe in self.self_pips]
    #     states = [step.state for step in steps]
    #     actions = [step.action for step in steps]
    #     n_states = [step.n_state for step in steps]
    #     dones = [step.done for step in steps]
    #     infos = [step.info for step in steps]
    #     rews = [step.rew for step in steps]
    #
    #     return states, act, rews, n_states, dones, infos

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
                flag = True
                break

            if step.info:
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

    def calc_drew(self, rews, last_state):  # last_state は時系列的に最後のstate (n_states[-1]でok)
        d_rews = []
        ac_rew, _ = self.network(last_state)
        ac_rew = ac_rew[0]
        for rew in rews:
            ac_rew = gamma * ac_rew + rew
            d_rews.append(ac_rew)
        d_rews.reverse()
        return  d_rews

    def init_state(self, worker_id):
        score = self.workers.score(worker_id)
        self.global_score.append(score)
        print("ID{} : score is {}. average is {}.".format(worker_id, score, float(sum(self.global_score)) / float(
            len(self.global_score))))

        self.w_states[worker_id] = self.workers.reset(worker_id)

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
        while True:
            states, d_rews, acts, probs = self.play_n_step()
            loss = self.network.calc_loss(states, acts, d_rews, probs)
            dot = torchviz.make_dot(loss, params=dict(self.network.named_parameters()))
            dot.format = 'png'
            global TMP
            if not TMP:
                dot.render('/home/emile/Documents/Code/breakout_A2C/graph_image')
                TMP = True
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()


def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()
