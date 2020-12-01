import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(torch.nn.Module):
    def __init__(self, action_space, step_repeat_times, alpha, beta):
        super(Network, self).__init__()
        # define network
        self.alpha = alpha
        self.beta = beta
        self.c1 = nn.Conv2d(step_repeat_times, 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        l5_out = 512
        self.l5 = nn.Linear(3136, l5_out)
        self.critic = nn.Linear(l5_out, 1)
        self.actor = nn.Linear(l5_out, action_space)

        # init bias of network
        nn.init.kaiming_normal_(self.c1.weight)
        nn.init.kaiming_normal_(self.c2.weight)
        nn.init.kaiming_normal_(self.c3.weight)
        nn.init.kaiming_normal_(self.l5.weight)
        nn.init.kaiming_normal_(self.critic.weight)
        nn.init.kaiming_normal_(self.actor.weight)
        self.train()

    def forward(self, inputs):  # c1 -> relu -> c2 -> relu -> c3 -> relu -> l5 -> relu -> actor/critic
        inputs = torch.from_numpy(inputs).float().to(dev)
        inputs /= 255.0
        x = F.relu(self.c1(inputs))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.view(1, -1)
        x = F.relu(self.l5(x))
        ans = self.critic(x), self.actor(x)
        return ans

    def select_action2(self, state):
        _, logits = self(state)
        probs = F.softmax(logits, dim=1).to(dev)
        act = probs.multinomial(num_samples=1).cpu()[0]
        return act.numpy().copy()

    def calc_loss2(self, states, acts, d_rews):
        probs, crt = [], []
        log_pis = []
        for state in states:
            tmp = self(state)
            probs.append(F.softmax(tmp[1], dim=1).to(dev)[0])
            log_pis.append(F.log_softmax(tmp[1], dim=1).to(dev)[0])
            crt.append(tmp[0][0])

        probs = torch.stack(probs, dim=0).to(dev)
        log_pis = torch.stack(log_pis, dim=0).to(dev)
        d_rews = torch.stack(d_rews, dim=0).view(-1).to(dev)
        v_states = torch.stack(crt, dim=0).view(-1).to(dev)
        acts = torch.from_numpy(np.array(acts, dtype=np.int64)).view(len(acts), 1).to(dev)
        pis = probs.gather(dim=1, index=acts).squeeze().to(dev)
        log_pis = log_pis.gather(1, acts).squeeze().to(dev)
        entropy = -torch.dot(pis, log_pis).to(dev)
        advantage = d_rews - v_states
        v_loss = (advantage ** 2).to(dev)
        a_loss = (log_pis * (advantage.detach())).to(dev)
        ans = (0.5 * self.alpha * v_loss - a_loss - self.beta * entropy).to(dev)
        ans = torch.sum(ans, dim=0).to(dev)
        return ans
