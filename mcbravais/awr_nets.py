import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from contextlib import contextmanager

from spacelib.flatter import Flatter

from .common_nets import (
    LoggingModule,
    HybridInputModule,
    mlp_layers
)

import pdb

class AWRAgent(object):
    default_params = {
            'latent_size': 800,

            'encoder_hidden': [800, 800],
            'decoder_hidden': [800, 800],

            'policy_trunk_hidden': [800, 800],

            'value_function_hidden': [800, 800],

            'ldm_net_hidden': [800,800],

            # 'gamma': 0.97,
            'beta': 500.0,
            }

    def __init__(self, observation_space, action_space, params={}):
        print(f"Params are {params}")
        self.params = {**self.default_params, **params}
        params = self.params
        self.obsflat = Flatter(observation_space)
        self.actflat = Flatter(action_space)
        self.observation_space = observation_space
        self.action_space = action_space

        self.state_encoder = Encoder(observation_space, params['latent_size'],params['encoder_hidden']) #
        self.encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=1.e-4)

        # self.state_decoder = Decoder(params['latent_size'],observation_space,params['decoder_hidden']) #
        # self.decoder_optimizer = torch.optim.Adam(self.state_decoder.parameters(), lr=1.e-4)

        self.policy = AWRPolicy(params['latent_size'], action_space, params['policy_trunk_hidden'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1.e-4)

        self.state_value = StateValueFunction(params['latent_size'], params['value_function_hidden']) #
        self.state_value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr=1.e-4)

        # self.reconstruction_loss = SpaceLoss(self.obsflat.flat_space)

        self.device = torch.device('cpu')


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

    def save_models(self, save_path, whitelist=None):
        nets = {k:v for k,v in self.__dict__.items() if isinstance(v, nn.Module)}
        if whitelist is not None:
            nets = {k:v for k,v in nets if k in whitelist}

        os.makedirs(save_path, exist_ok=True)
        for net_name, net in nets.items():
            torch.save(net.state_dict(), os.path.join(save_path, f'{net_name}.bin'))
            pickle.dump(net, open(os.path.join(save_path, f'{net_name}.p'), 'wb'))
        saved_list = '\n\t/' + '\n\t/'.join(nets.keys())
        print(f"Saved module states:\n {save_path} {saved_list}")

    def load_models(self, save_path, whitelist=None):
        models_to_load = [os.path.splitext(os.path.split(f)[-1])[0] for f in glob.glob(os.path.join(save_path,'*.bin'))]
        if whitelist is not None:
            models_to_load = [m for m in models_to_load if m in whitelist]

        for m in models_to_load:
            model_path = os.path.join(save_path, f'{m}.bin')
            mod_obj = pickle.load(open(os.path.join(save_path, f'{m}.p'),'rb'))
            mod_obj.load_state_dict(torch.load(model_path))
            self.__dict__[m] = mod_obj
        loaded_list  = '\n\t/' + '\n\t/'.join(models_to_load)
        print(f"Loaded module states:\n {save_path} {loaded_list}")


    def get_action(self, obs, reparameterize=True):
        with torch.no_grad(), self.state_encoder.to_eval(), self.policy.to_eval():
            hidden = self.state_encoder(self.obsflat.flatten(obs))
            # try:
            act, mu, lp = self.policy(hidden)
            # except:
            #     pdb.set_trace()
            
            if reparameterize:
                return self.policy.interpret(act)
            else:
                return self.policy.interpret(mu)

    @property
    def netdict(self):
        return {k:v for k,v in self.__dict__.items() if isinstance(v, nn.Module)}

    @property
    def has_nulls(self):
        return any([not torch.isfinite(x).all() for k,v in self.__dict__.items() if isinstance(v, nn.Module) for x in v.parameters()])

    def update_value(self, transition_with_dfr, writer = None):
        s, a, r, s_next, d, dfr = transition_with_dfr

        s_encoded = self.state_encoder(s)
        if not torch.isfinite(s_encoded).all():
            pdb.set_trace()
        s_values = self.state_value(s_encoded)
        v_loss = F.mse_loss(s_values, dfr)

        # autoencoder_loss = self.reconstruction_loss(s, self.state_decoder(s_encoded))

        self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        self.state_value_optimizer.zero_grad()
        # (v_loss + autoencoder_loss).backward()
        v_loss.backward()
        self.encoder_optimizer.step()
        self.state_value_optimizer.step()

        if writer: 
            # writer.add_scalar('update/autoencoder_loss', autoencoder_loss.detach())
            writer.add_scalar('update/state_value_loss', v_loss.detach())
            writer.add_histogram('update/predicted_state_values', s_values)

    def update_policy(self, transition_with_dfr, writer = None):
        s, a, r, s_next, d, dfr = transition_with_dfr
        s_encoded2 = self.state_encoder(s)
        s_values2 = self.state_value(s_encoded2.detach()).clamp(-5e3,5e3)

        action_log_probs = self.policy.log_prob(s_encoded2, a, writer=writer)
        action_advantages = (dfr - s_values2)
        policy_loss_tmp = - action_log_probs * torch.exp(action_advantages/self.params['beta'])
        policy_loss = policy_loss_tmp.clamp(-1e4,1e4).mean()

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
        self.encoder_optimizer.step()
        self.policy_optimizer.step()


        if writer:
            # sample_actions, sample_mus,_,_ = self.actor(s_encoded2)
            # writer.add_histogram('update/action_camera_norm', sample_actions[2].norm(dim=-1))
            # writer.add_histogram('update/action_camera_mu_norm', sample_mus[2].norm(dim=-1))
            writer.add_scalar('update/policy_loss', policy_loss.detach())
            writer.add_histogram('update/action_log_probs', action_log_probs.detach())
            writer.add_histogram('update/action_advantages', action_advantages.detach())


       

class StateValueFunction(LoggingModule):
    def __init__(self, latent_size, hidden_layer_sizes = []):
        super().__init__()
        self.latent_size = latent_size
        layer_sizes = [latent_size, *hidden_layer_sizes, 1]
        self.layers = nn.Sequential(*mlp_layers(layer_sizes, batch_norm=False))

    def forward(self, latent):
        res = self.layers(latent)
        assert res.shape[-1] == 1
        return res[...,0]


class Encoder(nn.Module):
    def __init__(self, observation_space, latent_size, hidden_layer_sizes = []):
        super().__init__()
        self.in_net = HybridInputModule(observation_space)
        layer_sizes = [self.in_net.n, *hidden_layer_sizes, latent_size]
        self.hidden_layers = nn.Sequential(*mlp_layers(layer_sizes, batch_norm=False), nn.LayerNorm(layer_sizes[-1]), nn.Tanh())
        self.n = layer_sizes[-1]
    
    @contextmanager
    def to_eval(self):
        tmp = self.training
        self.eval()
        yield self
        if tmp:
            self.train()

    def forward(self, X):
        X_in = X
        X = self.in_net(X)
        if X.dim() == 1:
            out_shape = (self.n, )
            X = X.unsqueeze(0) # for batch norm
        else:
            out_shape = (-1, self.n)
        if X.shape[0] == 1:
            with self.to_eval():
                res = self.hidden_layers(X).reshape(out_shape).contiguous()
        else:
            res = self.hidden_layers(X).reshape(out_shape).contiguous()
        return res

class BoxPolicy(LoggingModule):
    def __init__(self, latent_size, action_space, hidden_layer_sizes=[], log_std_bounds = (-8,-4)):
        super().__init__()
        self.space = action_space
        if not isinstance(action_space, gym.spaces.Box): raise ValueError
        if not np.isfinite([action_space.low, action_space.high]).all(): raise ValueError
        self.register_buffer('low', torch.tensor(action_space.low, dtype=torch.float32))
        self.register_buffer('high', torch.tensor(action_space.high, dtype=torch.float32))
        self.output_shape = action_space.shape
        self.latent_size = latent_size
        self.n = np.prod(action_space.shape)

        self.mu_net = nn.Sequential(*mlp_layers([latent_size, *hidden_layer_sizes, self.n]))
        self.log_std_net = nn.Sequential(*mlp_layers([latent_size, *hidden_layer_sizes, self.n]))

        self.log_std = nn.Linear(latent_size, np.prod(action_space.shape))
        self.reparameterize = True
        self.log_std_bounds = log_std_bounds

    def log_p(self, state, action, writer=None):
        # shape = (*action.shape[:-1], *self.output_shape)
        flat_shape = (*action.shape[:-len(self.output_shape)], self.n)
        has_batch_dim = state.dim() == 2
        # scale from (low, high) to (-1,1)
        rescaled_action = (((action - self.low)/(self.high - self.low))*2.0-1.0
                    ).clamp(-0.9999,0.9999
                    ).reshape(*flat_shape).contiguous()
        # state = state.reshape(-1, self.latent_size)
        # assert state.shape[0] == rescaled_action.shape[0]
        # Apply arctanh
        rescaled_action_ath = 0.5 * torch.log( (1+rescaled_action)/(1-rescaled_action) )

        mu = self.mu_net(state)

        lstd = self.log_std_net(state)
        lstd = self.log_std_bounds[0] + 0.5 * (self.log_std_bounds[1] - self.log_std_bounds[0]) * (torch.tanh(lstd) + 1)
        log_p = torch.sum(- 0.5*(((rescaled_action_ath - mu)/ (torch.exp(lstd) + 1.e-6))**2 + 2*lstd + np.log(2*np.pi)), axis=-1)
        log_p -= torch.log(F.relu(1 - rescaled_action.pow(2)) + 1.e-9).sum(axis=-1)

        if writer:
            writer.add_scalar('policy/log_std_mean', lstd.mean())

        # if has_batch_dim:
        #     return log_p.reshape(-1, 1)
        # else:
        return log_p
        

    def forward(self, X):
        # if X.dim() == 1:
        #     output_shape = self.output_shape
        #     scalar_output_shape = (-1, )
        # elif X.dim() == 2:
        #     output_shape = (-1, *self.output_shape)
        #     scalar_output_shape = (-1, 1)
        # else:
        #     raise ValueError(f"BoxPolicy for shape {self.output_shape} couldn't cope with input of shape {X.shape}.")
        mu = self.mu_net(X)
        lstd = self.log_std_net(X)
        lstd = self.log_std_bounds[0] + 0.5 * (self.log_std_bounds[1] - self.log_std_bounds[0]) * (torch.tanh(lstd) + 1.0)
        noise = torch.randn_like(mu)

        X = mu + noise * torch.exp(lstd)
        # Gaussian log p
        log_p = torch.sum(- 0.5*(((X - mu)/ (torch.exp(lstd) + 1.e-6))**2 + 2*lstd + np.log(2*np.pi)), axis=-1)

        # Formulation #2
        # residual = (-0.5 * noise.pow(2) - lstd).sum(-1, keepdim=True)
        # log_p = residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

        X = torch.tanh(X)
        mu = torch.tanh(mu)

        log_p_adjustment = torch.log(F.relu(1 - X.pow(2)) + 1.e-9).sum(axis=-1)
        log_p -= log_p_adjustment
        # X = X.reshape(output_shape).contiguous()
        # mu = mu.reshape(output_shape).contiguous()
        # log_p = log_p.reshape(scalar_output_shape)
        # lstd = lstd.reshape(output_shape).contiguous()

        X = (X + 1.0)*(self.high - self.low)/2.0 + self.low
        mu = (mu+ 1.0)*(self.high - self.low)/2.0 + self.low

        return X, mu, log_p


class DiscretePolicy(LoggingModule):
    def __init__(self, latent_size, action_space, hidden_layer_sizes=[], log_temp_bounds = (-12,2)):
        super().__init__()
        if not isinstance(action_space, gym.spaces.Discrete): raise ValueError

        self.n = action_space.n
        self.logit_net = nn.Sequential(*mlp_layers([latent_size, *hidden_layer_sizes, self.n]))
        self.latent_size = latent_size

    def log_p(self, state, action, writer=None):
        has_batch = action.dim()==2
        if action.shape[-1] != self.n:
            raise ValueError("DiscretePolicy log_p actions must be one-hot encoded.")
        if state.dim() != action.dim():
            raise ValueError("DiscretePolicy log_p expects state + actions to have the same dim()")
        # action = action.reshape(-1, self.n).contiguous()
        # state = state.reshape(-1, self.latent_size).contiguous()
        logits = self.logit_net(state).contiguous()

        dist = torch.distributions.Categorical(logits=logits)

        if action.shape[-1] == self.n:
            assert (action.sum(-1) == 1).all()
            action = action.argmax(-1).long()

        return dist.log_prob(action)
        # if action.dim() == 2:
        #     if action.shape[-1] > 1:
        #     if not (action.long() == action).all():
        #         print("Actions should be integer (or one hot encoded)")
        #         pdb.set_trace()
        #     action = action.long()

        # pdb.set_trace()
        # if has_batch:
        #     return dist.log_prob(action).reshape(-1,1)
        # else:
        #     return dist.log_prob(action)

    def forward(self, X):
        logits = self.logit_net(X)
        if logits.dim() == 2:
            output_shape = (-1, self.n)
            scalar_output_shape = (-1, 1)
        elif logits.dim() == 1:
            output_shape = (self.n,)
            scalar_output_shape = (1,)
        else:
            raise ValueError
        
        dist = torch.distributions.Categorical(logits=logits)
        samp = dist.sample()
        samp_onehot = F.one_hot(samp, self.n)

        return (
            samp_onehot.reshape(output_shape).contiguous(),
            logits.reshape(output_shape).contiguous(),
            dist.log_prob(samp).reshape(scalar_output_shape).contiguous()
        )
            
        
class AWRPolicy(LoggingModule):
    def __init__(self, latent_size, action_space, hidden_layer_sizes=[]):
        super().__init__()

        self.actflat = Flatter(action_space)

        layer_sizes = [latent_size, *hidden_layer_sizes]
        self.hidden_layers = nn.Sequential(*mlp_layers(layer_sizes), nn.LeakyReLU())

        self.policy_heads = nn.ModuleList()
        for space in self.actflat:
            if isinstance(space, gym.spaces.Box):
                tmp = max(10, np.prod(space.shape)*2)
                self.policy_heads.append(BoxPolicy(layer_sizes[-1], space, hidden_layer_sizes=[tmp]))
            elif isinstance(space, gym.spaces.Discrete):
                tmp = max(10, space.n*2)
                self.policy_heads.append(DiscretePolicy(layer_sizes[-1], space, hidden_layer_sizes=[tmp]))
            else:
                raise ValueError
        self.should_interpret = False

    def log_p(self, state, action, writer=None):
        X = self.hidden_layers(state)
        log_ps = [pol.log_p(X, act, writer=writer) for pol, act in zip(self.policy_heads, action)]
        if writer:
            for k,log_p in enumerate(log_ps):
                writer.add_scalar(f'policy_lp/log_p_{k}', log_p.detach().abs().mean().cpu())
        return sum(log_ps)

    def forward(self, X):
        X = self.hidden_layers(X)
        outputs, mus, log_ps = tuple(zip(*[pol(X) for pol in self.policy_heads]))
        for k,lp in enumerate(log_ps):
            try:
                self.log('add_histogram', f'policy/log_p_{k}', lp)
            except:
                pdb.set_trace()
        if self.should_interpret:
            return interpet(outputs)
        return outputs, mus, sum(log_ps)


    @contextmanager
    def interpret_actions(self):
        tmp = self.should_interpret
        self.should_interpret = True
        yield self
        self.should_interpret = tmp
    
    def interpret(self, action):
        outputs = [torch.argmax(o, dim=-1).detach().cpu() if isinstance(space, gym.spaces.Discrete) else o.squeeze().detach().cpu()
                    for o, space in zip(action, self.actflat)]
        return self.actflat.unflatten([o.item() if o.dim()==0 else o.numpy() for o in outputs])