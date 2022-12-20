import math
import pickle
from collections import deque
from os.path import exists
from subprocess import run

import cv2 as cv
import numpy as np
import scipy
import torch
from torch.nn import DataParallel, Module

from app.config import Config
from app.models import InceptionI3d


def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(
            img.squeeze(), [oheight, owidth], interp=interp, mode="F"
        ).reshape((oheight, owidth, chn))
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode='F')
            # resized_img[:, :, t] = np.array(Image.fromarray(img[:, :, t]).resize([oheight, owidth]))
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(
                img[:, :, t], [oheight, owidth]
            )
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv.resize(frame, (owidth, oheight)).reshape(
                oheight, owidth, in_chn
            )
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(
            resized_img.shape[0], resized_img.shape[1], chn
        )

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img


def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting (resp. dividing) by the mean (resp.
    std. deviation) statistics of a dataset in RGB space.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x


def to_torch(ndarray):

    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray

def to_numpy(tensor):

    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor

def im_to_numpy(img):
    
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img

def im_to_torch(img):

    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def load_model(checkpoint_path: str, number_of_classes: int, number_of_frames: int) -> Module:

    model = DataParallel(InceptionI3d(
        number_of_classes, 
        spatiotemporal_squeeze=True, 
        final_endpoint='Logits', 
        name="inception_i3d", 
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=number_of_frames
    )).cuda()

    if not exists(Config.checkpoint_path):
        run(['cat', 'app/checkpoints/*', '>>', Config.checkpoint_path])

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def load_vocabulary(vocabulary_path: str) -> dict:

    with open(vocabulary_path, 'rb') as file:
        return pickle.load(file)


def prepare_input(video: np.ndarray, input_resolution: int=224, resize_resolution: int=256, mean: torch.Tensor=0.5*torch.ones(3), std: torch.Tensor=1.0*torch.ones(3)) -> np.ndarray:
    video_tensor = torch.stack(
        [im_to_torch(frame[:, :, [2, 1, 0]]) for frame in list(video)]
    ).permute(1, 0, 2, 3)

    iC, iF, _, _ = video_tensor.shape
    video_tensor_resized = np.zeros((iF, resize_resolution, resize_resolution, iC))
    for t in range(iF):
        tmp = video_tensor[:, t, :, :]
        tmp = resize_generic(
            im_to_numpy(tmp), resize_resolution, resize_resolution, interp="bilinear", is_flow=False
        )
        video_tensor_resized[t] = tmp
    video_tensor_resized = np.transpose(video_tensor_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_resolution - input_resolution) / 2)
    uly = int((resize_resolution - input_resolution) / 2)
    # Crop 256x256
    video_tensor_resized = video_tensor_resized[:, :, uly : uly + input_resolution, ulx : ulx + input_resolution]
    video_tensor_resized = to_torch(video_tensor_resized).float()
    assert video_tensor_resized.max() <= 1
    video_tensor_resized = color_normalize(video_tensor_resized, mean, std)
    return video_tensor_resized

def sliding_windows(input_video: torch.Tensor, number_of_frames: int, stride: int) -> torch.Tensor:

    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = input_video.shape
    print(f"Input video shape: {input_video.shape}")
    # If needed, pad to the minimum clip length
    if nFrames < number_of_frames:
        rgb_ = torch.zeros(C, number_of_frames, H, W)
        rgb_[:, :nFrames] = input_video
        rgb_[:, nFrames:] = input_video[:, -1].unsqueeze(1)
        input_video = rgb_
        nFrames = input_video.shape[1]

    num_clips = math.ceil((nFrames - number_of_frames) / stride) + 1

    rgb_slided = torch.zeros(num_clips, 3, number_of_frames, H, W)
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(number_of_frames, nFrames - j * stride)
        if actual_clip_length == number_of_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - number_of_frames
        rgb_slided[j] = input_video[:, t_beg : t_beg + number_of_frames, :, :]
    return rgb_slided


def video_to_asl(video: deque, confidence: float, model, word_data) -> str:
    input_video = prepare_input(video)
    input_sliding_window = sliding_windows(input_video, Config.number_of_frames, Config.stride)

    num_clips = input_sliding_window.shape[0]
    # Group the clips into batches
    num_batches = math.ceil(num_clips / Config.batch_size)
    raw_scores = np.empty((0, Config.number_of_classes), dtype=float)
    for b in range(num_batches):
        inp = input_sliding_window[b * Config.batch_size : (b + 1) * Config.batch_size]
        # Forward pass
        out = model(inp)
        raw_scores = np.append(raw_scores, out["logits"].cpu().detach().numpy(), axis=0)
    prob_scores = scipy.special.softmax(raw_scores, axis=1)
    prob_sorted = np.sort(prob_scores, axis=1)[:, ::-1]
    pred_sorted = np.argsort(prob_scores, axis=1)[:, ::-1]

    word_topk = np.empty((Config.topk, num_clips), dtype=object)
    for k in range(Config.topk):
        for i, p in enumerate(pred_sorted[:, k]):
            word_topk[k, i] = word_data["words"][p]
    prob_topk = prob_sorted[:, :Config.topk].transpose()
    print(prob_topk[0,0])
    if prob_topk[0,0] > confidence:   
        return word_topk[0][0]
    # print(prob_topk)
    # print("Predicted signs:")
    # print(word_topk)

