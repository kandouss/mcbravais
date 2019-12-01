import datetime
import torch
import numpy as np
import moviepy.editor as mpy

def ensure_dir(path):
    path = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    os.makedirs(path, exist_ok=True)
    return path
    
def time_string():
    return datetime.datetime.now().strftime('%y.%m.%d_%H:%M:%S')

def export_video(X, outfile, fps=30, rescale_factor=2):
    frame_images = []
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if isinstance(X, np.float):
        X = (X*255).astype(np.uint8).clip(0,255)
    
    X = np.kron(X, np.ones((1, rescale_factor, rescale_factor, 1)))
    getframe = lambda t: X[int(t*fps)]
    clip = mpy.VideoClip(getframe, duration=(len(X)-1)/fps)
    clip.write_videofile(outfile, fps=fps)