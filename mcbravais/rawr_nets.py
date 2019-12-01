import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from contextlib import contextmanager

from spacelib.flatter import Flatter
from .awr_nets import (
    AWRPolicy,
    StateValueFunction
)

from .common_nets import (
    LoggingModule,
    HybridInputModule,
    mlp_layers
)

import pdb

def to_tensors(array_list, dtype, device):
    try:
        return [torch.from_numpy(a).to(dtype).to(device) for a in array_list]
    except:
        pdb.set_trace()

class RecurrentAWRAgent(object):
    default_params = {
            'latent_size': 256,
            'encoder_trunk': [],
            'lstm_hidden_size': 512,
            'policy_trunk_hidden': [512, 512, 512],
            'value_function_hidden': [512, 512, 512],
            'beta': 1.0,
            'learning_rate': 1.e-3
            }

    def __init__(self, observation_space, action_space, params={}):
        print(f"Params are {params}")
        self.params = {**self.default_params, **params}
        params = self.params
        self.obsflat = Flatter(observation_space)
        self.actflat = Flatter(action_space)
        self.observation_space = observation_space
        self.action_space = action_space

        self.state_encoder = LSTMEncoder(
            observation_space = observation_space,
            latent_size = params['latent_size'],
            lstm_hidden_size = params['lstm_hidden_size'],
            trunk_layers = params['encoder_trunk'])
        self.encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=self.params['learning_rate'])

        self.policy = AWRPolicy(params['latent_size'], action_space, params['policy_trunk_hidden'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.params['learning_rate'])

        self.state_value = StateValueFunction(params['latent_size'], params['value_function_hidden']) #
        self.state_value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr=self.params['learning_rate'])

        self.device = torch.device('cpu')

        self.hx = None

    def set_device(self, device):
        nets = [v for k,v in self.__dict__.items() if isinstance(v, nn.Module)]
        for net in nets:
            net.to(device)
        self.device = device

    @contextmanager
    def torch_device(self, device):
        nets = [v for k,v in self.__dict__.items() if isinstance(v, nn.Module)]
        old_device = self.device
        self.set_device(device)
        yield self
        self.set_device(old_device)

    def save_models(self, save_path, whitelist=None, mod_pickle=False):
        nets = {k:v for k,v in self.__dict__.items() if isinstance(v, nn.Module)}
        if whitelist is not None:
            nets = {k:v for k,v in nets if k in whitelist}

        os.makedirs(save_path, exist_ok=True)
        for net_name, net in nets.items():
            torch.save(net.state_dict(), os.path.join(save_path, f'{net_name}.bin'))
            if mod_pickle:
                pickle.dump(net, open(os.path.join(save_path, f'{net_name}.p'), 'wb'))
            # net.save(f'{net_name}.p')
        saved_list = '\n\t/' + '\n\t/'.join(nets.keys())
        print(f"Saved module states:\n {save_path} {saved_list}")

    def load_models(self, save_path, whitelist=None, mod_pickle=False):
        models_to_load = [os.path.splitext(os.path.split(f)[-1])[0] for f in glob.glob(os.path.join(save_path,'*.bin'))]
        if whitelist is not None:
            models_to_load = [m for m in models_to_load if m in whitelist]

        for m in models_to_load:
            model_path = os.path.join(save_path, f'{m}.bin')
            if mod_pickle:
                mod_obj = pickle.load(open(os.path.join(save_path, f'{m}.p'),'rb'))
                mod_obj.load_state_dict(torch.load(model_path))
                self.__dict__[m] = mod_obj
            else:
                self.__dict__[m].load_state_dict(torch.load(model_path))
        loaded_list  = '\n\t/' + '\n\t/'.join(models_to_load)
        print(f"Loaded module states:\n {save_path} {loaded_list}")

    def end_episode(self):
        self.hx = None

    def get_action(self, obs, update_hx=True, sample=True):
        with torch.no_grad(), self.state_encoder.to_eval(), self.policy.to_eval():
            o = [np.atleast_1d(x).copy() for x in self.obsflat.flatten(obs)]
            obs_tensor = to_tensors(o, dtype=torch.float32, device=self.device)
            latent, hx = self.state_encoder(obs_tensor, hx=self.hx, return_hidden=True)
            self.hx = hx

            act, mu, lp = self.policy(latent)

            if sample:
                return self.policy.interpret(act)
            else:
                return self.policy.interpret(mu)

    @property
    def netdict(self):
        return {k:v for k,v in self.__dict__.items() if isinstance(v, nn.Module)}

    @property
    def has_nulls(self):
        return any([not torch.isfinite(x).all() for k,v in self.__dict__.items() if isinstance(v, nn.Module) for x in v.parameters()])

    # def refresh_hidden_cache(self, sequence_iterator, set_hidden_callback):
    #     with torch.no_grad():
    #         for k, (seq, start_hidden, idx) in enumerate(sequence_iterator):
    #             set_hidden_callback(
    #                 *idx,
    #                 self.state_encoder.get_hidden(seq[0], start_hidden)
    #             )

    def iter_update_combined(self, minibatch_iter, warmup_k=0, writer=None):
        
        w_abs_max = 1.e4

        batch_policy_loss = 0.0
        batch_value_loss = 0.0
        w_clip_count = 0

        torch.autograd.set_detect_anomaly(True)

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.state_value_optimizer.zero_grad()

        value_loss = 0

        action_log_p = []
        action_advantages = []

        for minibatch_no, mbatch in enumerate(minibatch_iter):
            true_values = mbatch.val

            state_latent = self.state_encoder(mbatch.obs, return_hidden=False, check_grad=True)
            state_values = self.state_value(state_latent).clamp(-5e4, 5e4)
            state_values = state_values
            assert true_values.shape == state_values.shape

            value_loss += F.mse_loss(true_values[:, warmup_k:], state_values[:, warmup_k:])
            batch_value_loss += value_loss.detach().cpu().item()

            action_log_p.append(self.policy.log_p(state_latent, mbatch.act, writer=writer)[:, warmup_k:])
            action_advantages.append((true_values - state_values.detach())[:, warmup_k:])
            assert action_advantages[-1].shape == action_log_p[-1].shape

        action_advantages = torch.cat(action_advantages).flatten()
        action_log_p = torch.cat(action_log_p).flatten()
        
        adv_mask = action_advantages < (self.params['beta']*np.log(w_abs_max))
        action_advantages = action_advantages[adv_mask]
        action_log_p = action_log_p[adv_mask]
        
        action_advantages_normed = ((action_advantages - action_advantages.mean())/
                             (action_advantages.std()+1.e-6))

        exp_advantages = torch.exp(action_advantages_normed / self.params['beta'])
        policy_loss = - (action_log_p * exp_advantages).mean()
        
        total_loss = (policy_loss + value_loss)
        batch_policy_loss = policy_loss.detach().cpu().item()

        w_clip_count += (exp_advantages.detach().abs()>=w_abs_max).sum()

        total_loss = (value_loss + policy_loss)
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 100)
        nn.utils.clip_grad_norm_(self.policy.parameters(), 100)

        self.encoder_optimizer.step()
        self.policy_optimizer.step()
        self.state_value_optimizer.step()

        if writer:
            writer.add_scalar('update/batch_policy_loss', batch_policy_loss/(minibatch_no + 1), period=1)
            writer.add_scalar('update/batch_value_loss', batch_value_loss/(minibatch_no + 1), period=1)
            writer.add_scalar('update/policy_loss_variety', exp_advantages.detach().std().cpu(), period=1)
            writer.add_scalar('update/w_clip_count', sum(~adv_mask)/len(adv_mask), period=1)


    # def iter_update_value(self, sequence_iter, writer = None):

    #     # with torch.autograd.set_detect_anomaly(True):
    #     self.encoder_optimizer.zero_grad()
    #     self.state_value_optimizer.zero_grad()

    #     for seq_no, (sequence,_,_) in enumerate(sequence_iter):
    #         true_values = sequence[-1]
    #         pred_values = self.state_value(self.state_encoder(sequence[0], return_hidden=False))
    #         value_loss = F.mse_loss(true_values.squeeze(), pred_values.squeeze())
    #         value_loss.backward()

    #     nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 10)
    #     self.encoder_optimizer.step()
    #     self.state_value_optimizer.step()

    #     if writer:
    #         writer.add_scalar('update/value_loss', value_loss.detach())

    # def iter_update_policy(self, sequence_iter, writer = None):

    #     self.encoder_optimizer.zero_grad()
    #     self.policy_optimizer.zero_grad()

    #     for seq_no, (sequence,_,_)  in enumerate(sequence_iter):
    #         true_values = sequence[-1]
    #         state_latent = self.state_encoder(sequence[0], return_hidden=False)
    #         state_values = self.state_value(state_latent.detach()).clamp(-5e3,5e3).detach()
    #         action_log_p = self.policy.log_p(obs_latent, sequence[1], writer=writer)
    #         action_advantages = (true_values - state_values)
    #         policy_loss = - (action_log_p * torch.exp(action_advantages/self.params['beta']))
    #         policy_loss.backward()

    #     nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 10)

    #     self.encoder_optimizer.step()
    #     self.policy_optimizer.step()

class JITLSTMCell(torch.jit.ScriptModule):
    __constants__ = ["n"]
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.register_buffer("dev", torch.tensor(0, dtype=torch.float32))
        self.n = hidden_size

    @torch.jit.script_method
    def _fwd(self, X, hidd, cell, hidd_out, cell_out):
        for k in range(X.shape[1]):
            hidd, cell = self.lstm_cell(X[:,k,:], (hidd, cell))
            hidd_out[:,k,:] = hidd
            cell_out[:,k,:] = cell
        return hidd, cell

    @torch.jit.script_method
    def _fwd_vec(self, X, hidd, cell):
        for k in range(X.shape[1]):
            hidd, cell = self.lstm_cell(X[:,k,:], (hidd, cell))
        return hidd, cell

    @torch.jit.ignore
    def forward(self, X, start_hx):
        out_shape = (*X.shape[:-1], self.n)
        if X.dim() == 1: # A single vector
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.dim() == 2: # A sequence
            X = X.unsqueeze(0)
        hidd_out = torch.zeros((X.shape[0], X.shape[1], self.n), dtype=torch.float32, device=self.dev.device)
        cell_out = torch.zeros((X.shape[0], X.shape[1], self.n), dtype=torch.float32, device=self.dev.device)
        if start_hx is None:
            start_hx = (torch.zeros_like(hidd_out[:,0,:]), torch.zeros_like(hidd_out[:,0,:]))
        elif start_hx[0].dim() == 1:
            start_hx = (start_hx[0].unsqueeze(0), start_hx[1].unsqueeze(0))
        self._fwd(X, start_hx[0], start_hx[1], hidd_out, cell_out)
        return hidd_out.reshape(out_shape), cell_out.reshape(out_shape)


class LSTMEncoder(LoggingModule):
    def __init__(self, observation_space, latent_size, lstm_hidden_size, trunk_layers=[]):
        super().__init__() 
        self.in_net = HybridInputModule(observation_space, frame_stack_depth = None)
        self.lstm = JITLSTMCell(input_size=self.in_net.n,hidden_size=lstm_hidden_size)

        self.trunk_layers = nn.Sequential(
            *mlp_layers([lstm_hidden_size + self.in_net.n, *trunk_layers, latent_size]),
            nn.LayerNorm(latent_size),
            nn.Tanh())
        self.n = latent_size

    def forward(self, X, hx=None, return_hidden=True, check_grad = False):
        X1 = self.in_net(X)
        hx = self.lstm(X1, hx)
        X_lstm = hx[0]
        if check_grad:
            assert X_lstm.requires_grad
        X_cat = torch.cat((X1, X_lstm), dim=-1)
        X_out = self.trunk_layers(X_cat)

        if return_hidden:
            return X_out, hx
        else:
            return X_out

    def get_hidden(self, X, hx=None):
        return self.lstm.get_hidden(self.in_net(X), hx)
