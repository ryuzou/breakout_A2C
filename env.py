import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import gym
from gym import wrappers
from PIL import Image

pic_width = 84
pic_height = 84


def change_pict(pict):
    # pict = pict[49:193, 8:152]  # new
    # ans = resize(rgb2gray(pict), (pic_width, pic_height))
    # return np.array(ans)
    frame = Image.fromarray(pict)
    frame = frame.convert("L")
    frame = frame.crop((0, 20, 160, 210))
    frame = frame.resize((pic_width, pic_height))
    frame = np.array(frame, dtype=np.float32)
    return frame


class gym_env:
    def __init__(self, worker_id):
        self.env = gym.make("BreakoutDeterministic-v4")
        self.env.seed = worker_id
        self.env.reset()
        if worker_id == 0:
            self.env = wrappers.Monitor(self.env, "/home/emile/Videos/", video_callable=(lambda ep: ep % 8 == 0))
        self.life = -1
        self.fire_action = 1

    def reset(self):
        state = self.env.reset()
        self.life = -1
        return change_pict(state)

    def step(self, action):  # state, rew, done, ライフの増減(Trueなら減ってる) を返す.
        n_state, rew, done, info = self.env.step(action)
        n_state = change_pict(n_state)
        if self.life == -1:  # info["ale.lives"]はlifeの数を意味する
            self.life = info["ale.lives"]
            return n_state, rew, done, False
        elif self.life != info["ale.lives"]:
            self.life = info["ale.lives"]
            return n_state, rew, done, True
        else:
            return n_state, rew, done, False
