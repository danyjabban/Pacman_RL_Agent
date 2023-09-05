import argparse
import logging
import timeit

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

import gymnasium as gym

import hw4_utils as utils
import matplotlib.pyplot as plt


logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


class ActorCriticNetwork(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size is 8960 assuming dims and convs above
        self.fc1 = nn.Linear(8960, args.hidden_dim)
        self.q = nn.Linear(args.hidden_dim, naction)
        self.v = nn.Linear(args.hidden_dim, 1)


    def forward(self, X, prev_state=None):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action logits, prev_state
        """
        bsz, T = X.size()[:2]

        Z = F.gelu(self.conv3( # bsz*T x hidden_dim x H3 x W3
              F.gelu(self.conv2(
                F.gelu(self.conv1(X.view(-1, self.iC, self.iH, self.iW)))))))

        # flatten with MLP
        Z = F.gelu(self.fc1(Z.view(bsz*T, -1))) # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)

        q = self.q(Z)
        v = self.v(Z)

        return v, q, prev_state
    
    def get_action(self, x, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        _, logits, prev_state = self(x, prev_state)
        # take highest scoring action
        action = logits.argmax(-1).squeeze().item()
        return action, prev_state
    

class DifferenceImage():
    def __init__(self, observations):
        if observations == None:
            self.obs = 0
        else:
            self.obs = observations

    def get_dif(self, new_obs):
        dif_obs = new_obs - self.obs
        self.obs = new_obs
        return dif_obs


def pg_step(stepidx, model, model_old, optimizer, scheduler, envs, observations, prev_state, dif_im, bsz=4, gamma=.99):
    if envs is None:
        envs = [gym.make(args.env) for _ in range(bsz)]
        observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
        observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
            [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)
        prev_state = None
        
    
        
    logits, rewards, actions, values, values_old_mod = [], [], [], [], []
    not_terminated = torch.ones(bsz) # agent is still alive
    for t in range(args.unroll_length):
        value_t, logits_t, prev_state = model(observations, prev_state) # logits are bsz x 1 x naction
        
        logits.append(logits_t)
        values.append(value_t.squeeze(1))
        
        # sample actions
        actions_t = Categorical(logits=logits_t.squeeze(1)).sample()
        actions.append(actions_t.view(-1, 1)) # bsz x 1
        # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
        env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
   
        rewards_t = torch.tensor([eo[1] for eo in env_outputs])
        # if we lose a life, zero out all subsequent rewards
        still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
        not_terminated.mul_(still_alive.float())
        rewards.append((rewards_t*not_terminated).view(-1,1))
        observations = torch.stack([utils.preprocess_observation(eo[0]) for eo in env_outputs]).unsqueeze(1)
        observations = dif_im.get_dif(observations) # got difference between obs_t and obs_t+1

        with torch.no_grad():
            value_old_t1, _, _ = model_old(observations, prev_state) # old model value t + 1
            values_old_mod.append(value_old_t1.squeeze(1))
        
    
    value_t, _, _ = model(observations, prev_state) # logits are bsz x 1 x naction
    values.append(value_t.squeeze(1))

    values = torch.cat(values, dim=1) # bsz x T
    values_t = values[:, :-1] # values at time t
    values_t1 = values[:, 1:] # values at time t + 1

    values_old_mod = torch.cat(values_old_mod, dim=1)


    
    rewards = torch.cat(rewards, dim=1)
    critic_loss = (values_t - rewards - gamma*values_old_mod).pow(2).mean()

    advantage = rewards + values_t - gamma*values_t1


    logits = torch.cat(logits, dim=1) # bsz x T x naction
    actions = torch.cat(actions, dim=1) # bsz x T 
    cross_entropy = F.cross_entropy(
        logits.view(-1, logits.size(2)), actions.view(-1), reduction='none')
    

    actor_loss = (cross_entropy.view_as(actions) * advantage).mean()
    
    total_loss = actor_loss + critic_loss

    stats = {"mean_return": sum(r.mean() for r in rewards)/len(rewards),
             "total_loss": total_loss.item()}
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
    optimizer.step()
    scheduler.step()

    # reset any environments that have ended
    for b in range(bsz):
        if not_terminated[b].item() == 0:
            obs = envs[b].reset(seed=stepidx+b)[0]
            observations[b].copy_(utils.preprocess_observation(obs))

    return stats, envs, observations, prev_state


def train(args):
    T = args.unroll_length
    B = args.batch_size

    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()
    del env

    model = ActorCriticNetwork(naction, args)
    model_old = ActorCriticNetwork(naction, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def lr_lambda(epoch): # multiplies learning rate by value returned; can be used to decay lr
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def checkpoint():
        if args.save_path is None:
            return
        
        save_path = f'model_state_bs:{args.batch_size}_lr:{args.learning_rate}_hd:{args.hidden_dim}_kw:{8}.pt'
        print('in checkpoint******************************')
        logging.info("Saving checkpoint to {}".format(save_path))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args}, save_path)


    timer = timeit.default_timer
    last_checkpoint_time = timer()
    envs, observations, prev_state = None, None, None
    # initialize difference image class with initial image
    dif_im = DifferenceImage(observations)
    frame = 0
    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame
        stats, envs, observations, prev_state = pg_step(
            frame, model, model_old, optimizer, scheduler, envs, observations, prev_state, dif_im, bsz=B)
        frame += T*B # here steps means number of observations
        prev_params = model.state_dict()
        model_old.load_state_dict(prev_params)

        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()
        
        if frame == args.total_frames:
            checkpoint()


        sps = (frame - start_frame) / (timer() - start_time)
        logging.info("Frame {:d} @ {:.1f} FPS: total_loss {:.3f} | mean_ret {:.3f}".format(
          frame, sps, stats['total_loss'], stats["mean_return"]))
        
        
        if frame > 0 and frame % (args.eval_every*T*B) == 0:
            print('in validate')
            utils.validate(model, render=args.render)
            model.train()


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--mode", default="train", choices=["train", "valid",], 
                    help="training or validation mode")
parser.add_argument("--total_frames", default=1000000, type=int, 
                    help="total environment frames to train for")
parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, 
                    help="unroll length (time dimension)")
parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")


if __name__ == "__main__":
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    if args.mode == "train":
        train(args)
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env        
        model = ActorCriticNetwork(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args)
