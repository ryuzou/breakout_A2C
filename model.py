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
        self.c1 = nn.Conv2d(step_repeat_times, 32, 8, stride=4, padding=1)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.c4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        l5_out = 512
        self.l5 = nn.Linear(6400, l5_out)  # 大きさ分からん
        self.critic = nn.Linear(l5_out, 1)
        self.actor = nn.Linear(l5_out, action_space)

        # init bias of network
        nn.init.kaiming_normal_(self.c1.weight)
        nn.init.kaiming_normal_(self.c2.weight)
        nn.init.kaiming_normal_(self.c3.weight)
#        nn.init.kaiming_normal_(self.c4.weight)
        nn.init.kaiming_normal_(self.critic.weight)
        nn.init.kaiming_normal_(self.actor.weight)
        # self.critic.bias.data.fill_(0)
        # self.actor.bias.data.fill_(0)
        self.train()


    def forward(self, inputs):  # c1 -> elu -> c2 -> elu -> c3 -> elu -> c4 -> elu -> flatten -> l5 -> elu -> actor/critic
        inputs = torch.from_numpy(inputs).float().to(dev)
        inputs /= 255.0
        # print("inputs : {}".format(inputs.shape))
        x = self.c1(inputs)
        # print("x shape : {}".format(x.shape))
        x = F.relu(x)
        # x = F.elu(self.c1(inputs))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        # x = F.elu(self.c4(x))  # [1, 32, 6, 6]
        x = x.view(1, -1)
        # print("x shape : {}".format(x.shape))
        x = F.relu(self.l5(x))
        # x = x.view(4 * 256)
        ans = self.critic(x), self.actor(x)
        # print("net critic shape : {}".format(self.critic(x).shape))
        # print("net actor shape : {}".format(self.actor(x).shape))
        return ans

    def select_action(self, state):
        _, logits = self(state)
        probs = F.softmax(logits, dim=0).to(dev)
        c_rand = torch.distributions.categorical.Categorical(probs=probs.detach())  # detach はTensorから勾配を抜いた物
        act = c_rand.sample().cpu()
        # print("act {}".format(act))
        return act.numpy().copy(), probs

    def select_action2(self, state):
        _, logits = self(state)
        probs = F.softmax(logits, dim=0).to(dev)
        c_rand = torch.distributions.categorical.Categorical(probs=probs.detach())  # detach はTensorから勾配を抜いた物
        act = c_rand.sample().cpu()
        # print("act {}".format(act))
        return act.numpy().copy()

    def set_weight_on_network(self, g_net):  # 変数をセット(グローバルネットから取得した変数)
        for param, g_param in zip(self.parameters(), g_net.parameters()):
            if g_param.data is not None:
                return
            param.data = g_param.data.clone()

    def calc_loss(self, states, acts, d_rews, probs):
        acts_one_hot = torch.from_numpy(np.identity(4)[acts]).to(dev)  # one_hot ベクトルに変換
        probs = torch.stack(probs, dim=0).to(dev)
        d_rews = torch.stack(d_rews, dim=0).view(-1).to(dev)
        tmp = [self(state)[0][0] for state in states]
        v_states = torch.stack(tmp, dim=0).view(-1).to(dev)

        # print("states size {}".format(v_states.shape))
        # # print("acts size {}".format(acts.shape))
        # print("d_rews size {}".format(d_rews.shape))
        # print("probs size {}".format(probs.shape))

        # pis = torch.dot(acts_one_hot, probs)  # pi(a_t|s_t) (t=1,2,...)
        pis = torch.sum(acts_one_hot * probs, dim=1).to(dev)
        # print("pis shape {}.".format(pis.shape))
        log_pis = torch.log(pis).to(dev)  # log(pi(a_t|s_t)) (t=1,2,...)

        entropy = -torch.dot(pis, log_pis).to(dev)
        # print("d_res shape {}".format(d_rews.shape))
        # print("v_state shape {}".format(v_states.shape))
        advantage = d_rews - v_states
        # print("adv shape {}".format(advantage.shape))
        # print("log_pis {}".format(log_pis.shape))
        v_loss = (advantage **2).to(dev)
        a_loss = (log_pis * advantage).to(dev)
        # print("a_loss shape {}".format(a_loss.shape))
        ans = 0.5*self.alpha*v_loss - a_loss - self.beta*entropy
        ans = ans.to(dev)
        ans = torch.sum(ans, dim=0).to(dev)

        return ans

    def calc_loss2(self, states, acts, d_rews):
        acts_one_hot = torch.from_numpy(np.identity(4)[acts]).to(dev)  # one_hot ベクトルに変換
        probs, crt = [], []
        for state in states:
            tmp = self(state)
            prob = F.softmax(tmp[1], dim=0).to(dev)
            probs.append(F.softmax(tmp[1], dim=0).to(dev)[0])
            crt.append(tmp[0][0])

        probs = torch.stack(probs, dim=0).to(dev)
        d_rews = torch.stack(d_rews, dim=0).view(-1).to(dev)
        v_states = torch.stack(crt, dim=0).view(-1).to(dev)

        pis = torch.sum(acts_one_hot * probs, dim=1).to(dev)
        log_pis = torch.log(pis).to(dev)  # log(pi(a_t|s_t)) (t=1,2,...)
        entropy = -torch.dot(pis, log_pis).to(dev)
        advantage = d_rews - v_states
        v_loss = (advantage **2).to(dev)
        a_loss = (log_pis * advantage).to(dev)
        ans = 0.5*self.alpha*v_loss - a_loss - self.beta*entropy
        ans = ans.to(dev)
        ans = torch.sum(ans, dim=0).to(dev)

        return ans
