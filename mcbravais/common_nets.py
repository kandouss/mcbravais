import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from contextlib import contextmanager

from spacelib.flatter import Flatter

import pdb


def mlp_layers(layer_sizes, batch_norm = False):
    layers = []
    linear_sizes = list(zip(layer_sizes, layer_sizes[1:]))
    for k, (m, m_) in enumerate(linear_sizes):
        layers.append(nn.Linear(int(m),int(m_)))
        if k < len(linear_sizes)-1:
            if batch_norm:
                layers.append(nn.BatchNorm1d(m_))
            layers.append(nn.LeakyReLU())
    return layers

class LoggingModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_writer = None

    @contextmanager
    def to_eval(self):
        tmp = self.training
        self.eval()
        yield self
        if tmp:
            self.train()

    @contextmanager
    def log_to(self, writer):
        old_writer = self.log_writer
        self.log_writer = writer
        yield self
        if old_writer is not None:
            self.log_writer = old_writer

    def log(self, log_method, tag, X):
        if self.log_writer is not None:
            log_hook = getattr(self.log_writer, log_method, None)
            if log_hook is None:
                raise ValueError(f"Log writer doesn't have method {log_method}.")
            log_hook(tag, X)


class HybridInputModule(LoggingModule):
    def __init__(self, space, frame_stack_depth=None):
        super().__init__()
        self.pre_hook = None
        self.flatter = Flatter(space)
        self.singleton = isinstance(space, (gym.spaces.Box, gym.spaces.Discrete))

        sample_input = list(self.flatter.sample())
        
        self.parallel_inputs = nn.ModuleList()
        for k,space in enumerate(self.flatter):
            if isinstance(space, gym.spaces.Box):
                if np.prod(space.shape) > 1000: # probably an image
                    if frame_stack_depth is not None:
                        assert isinstance(frame_stack_depth, int)
                        self.parallel_inputs.append(StackedImageInputModule(space, frame_stack_depth))
                        sample_input[k] = np.stack([sample_input[k]]*frame_stack_depth,axis=0)
                    else:
                        self.parallel_inputs.append(ImageInputModule(space))
                else:
                    self.parallel_inputs.append(BoxInputModule(space))
            elif isinstance(space, gym.spaces.Discrete):
                self.parallel_inputs.append(DiscreteInputModule(space))
            else:
                raise ValueError(f"Unsupported input space {space}")
        self.frame_stack_depth = frame_stack_depth
        self.n = self(sample_input).shape[0]
        assert self.n == sum(a.n for a in self.parallel_inputs)

    @contextmanager
    def flatten_inputs(self):
        self.pre_hook = self.flatter.flatten
        yield self
        self.pre_hook = None

    def forward(self, X):
        if self.pre_hook:
            X = self.pre_hook(X)
        # if self.singleton:
        #     if isinstance(X, tuple) and len(X) == 1:
        #         X = X[0]
        #     return self.parallel_inputs[0](X)

        results = []
        for mod_number, (mod, x) in enumerate(zip(self.parallel_inputs, X)):
            results.append(mod(x))
        try:
            return torch.cat(results, axis=-1)
        except:
            print("Failed to concatenate results.")
            pdb.set_trace()

class BoxInputModule(LoggingModule):
    def __init__(self, space):
        super().__init__()
        if not isinstance(space, gym.spaces.Box):
            raise ValueError(f"Attempted to make a BoxInputModule for a space of type {type(space)}")
        self.space = space
        self.n = int(np.prod(space.shape))
        self.register_buffer('placeholder', torch.tensor(0,dtype=torch.float32))

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            try:
                X = torch.tensor(X, dtype=torch.float32, device=self.placeholder.device)
            except:
                pdb.set_trace()
        if len(self.space.shape) == 0:
            if len(X.shape) == 0 or X.shape[-1] != 1:
                return X.unsqueeze(-1)
            else:
                return X
        elif (X.shape[-1] == self.n):
            return X
        elif X.shape[-len(self.space.shape):]==self.space.shape:
            tmp = (*X.shape[:-len(self.space.shape)], self.n)
            return X.reshape(tmp)
        else:
            raise ValueError(f"Confused about input with shape {tuple(X.shape)} for boxinputmodule from space {self.space.shape}")

class StackedImageInputModule(LoggingModule):
    def __init__(self, space, n_frames):
        super().__init__()
        assert space.shape in ((64,64,3),(3,64,64))
        self.space = space
        self.n_frames = n_frames
        self.image_encoder = ImageInputModule(space)
        self.n = self.n_frames * self.image_encoder.n

    def forward(self, X):
        # (batch, stack, channel, X, Y)
        # or
        # (batch, stack, X, Y, channel)
        
        if len(X.shape) == 4:
            X = X[None,:,:,:,:]
            output_shape = (self.n)
        else:
            output_shape = (-1, self.n)
        assert X.shape[1] == self.n_frames

        return torch.cat(
                [self.image_encoder(X[:,k,:,:,:]) for k in range(X.shape[1])],
                dim=-1).reshape(output_shape).contiguous()


class ImageInputModule(LoggingModule):
    def __init__(self, space):
        super().__init__()
        assert space.shape in ((64,64,3),(3,64,64))
        self.space = space
        shape = (3,64,64)

        self.conv_layers = nn.Sequential(
                # AffineColorTransform(shape[0], shape[0]),
                nn.Conv2d(3, 64, kernel_size=3, stride=1), # 62
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.Conv2d(64, 64, kernel_size=3, stride=1), # 60
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.MaxPool2d(2, 2), # 30

                nn.Conv2d(64, 96, kernel_size=3, stride=1), # 28
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.Conv2d(96, 96, kernel_size=3, stride=1), # 26
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.Conv2d(96, 96, kernel_size=3, stride=1), # 24
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.MaxPool2d(2, 2),
                nn.Conv2d(96, 128, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.Conv2d(128, 128, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),

                nn.Conv2d(128, 128, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),)
        self.n = int(np.prod(self.conv_layers(torch.zeros(1, shape[0],shape[1],shape[2])).shape))

    @staticmethod
    def preprocess_image(X, normalize_intensity = False):
        extra_dims = len(X.shape) - 3
        if X.shape[-3:] == (64,64,3):
            X = X.permute(*range(extra_dims), *(np.array([2,0,1])+extra_dims))
        
        if normalize_intensity:
            m_ = X.max(-1)[0].max(-1)[0].max(-1)[0].view(*X.shape[:extra_dims],1,1,1)
            m_[m_<1.e-10] = 1
        else:
            m_ = 255.0 if X.max() > 1 else 1.0

        return X/m_

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).contiguous().float().to(next(self.parameters()).device)
        output_shape = (*X.shape[:-3], self.n)

        X = self.preprocess_image(X)
        # if len(X.shape) == 3:
        #     X = X.unsqueeze(0)
        return self.conv_layers(X.reshape(-1,3,64,64)).reshape(output_shape).contiguous()


# class DiscreteInputModule(LoggingModule):
#     def __init__(self, space):
#         super().__init__()
#         self.n = space.n
#         self.register_buffer('eye', torch.eye(self.n, dtype=torch.float32))

#     def forward(self, X):
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X, device=self.eye.device).contiguous()
#         if X.dim() == 0:
#             return self.eye[X.long()].reshape(self.n)
#         if X.dim() == 1:
#             warnings.warn(
#                 "DiscreteInputModule input should probably have an explicit batch dimension.",
#                 stacklevel=2
#             )
#             if (X.shape[0] > self.n) or (X.max() > 1):
#                 return self.eye[X.long()].reshape(-1, self.n).contiguous()
#             if X.shape[0] == self.n:
#                 return X.float().reshape(self.n).contiguous()
#             else:
#                 raise ValueError(f"Didn't expect input of shape {X.shape} for discrete({self.n}) space.")
#         elif X.dim() == 2:
#             if X.shape[-1] == 1:
#                 return self.eye[X.long()].reshape(-1, self.n).contiguous()
#             elif X.shape[-1] == self.n:
#                 return X.float().reshape(-1, self.n).contiguous()
#             else:
#                 raise ValueError(f"Didn't expect input of shape {X.shape} for discrete({self.n}) space.")
#         else:
#             raise ValueError(f"Didn't expect input of shape {X.shape} for discrete({self.n}) space.")
class DiscreteInputModule(LoggingModule):
    def __init__(self, space):
        super().__init__()
        self.n = space.n
        self.register_buffer('eye', torch.eye(self.n, dtype=torch.float32))

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.eye.device).contiguous()
        if( len(X.shape) > 0) and (X.shape[-1] == self.n):
            return X.float()
        else:
            return torch.nn.functional.one_hot(X, num_classes=self.n).float()

        if X.dim() == 0:
            return self.eye[X.long()].reshape(self.n)
        if X.dim() == 1:
            warnings.warn(
                "DiscreteInputModule input should probably have an explicit batch dimension.",
                stacklevel=2
            )
            if (X.shape[0] > self.n) or (X.max() > 1):
                return self.eye[X.long()].reshape(-1, self.n).contiguous()
            if X.shape[0] == self.n:
                return X.float().reshape(self.n).contiguous()
            else:
                raise ValueError(f"Didn't expect input of shape {X.shape} for discrete({self.n}) space.")
        elif X.dim() == 2:
            if X.shape[-1] == 1:
                return self.eye[X.long()].reshape(-1, self.n).contiguous()
            elif X.shape[-1] == self.n:
                return X.float().reshape(-1, self.n).contiguous()
            else:
                raise ValueError(f"Didn't expect input of shape {X.shape} for discrete({self.n}) space.")
        else:
            raise ValueError(f"Didn't expect input of shape {X.shape} for discrete({self.n}) space.")

def geoseq(start, stop, n_layers):
    k = np.log(stop/start)/(n_layers-1)
    return [start, *[int(start*np.exp(k*n)) for n in range(1, n_layers-1)], stop]