import torch
import json
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Dict, List, Sequence
from transformers import PreTrainedTokenizer
import open3d as o3d
import random
import os
import sys
import numpy as np
from torch.backends import cudnn
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 

def resize_and_center_crop(img, target_size):
    # 获取原始宽高
    original_width, original_height = img.size
    target_width, target_height = target_size, target_size

    # 计算调整大小时的比例
    scale = max(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # 计算中心裁剪的区域
    left = (new_size[0] - target_width) // 2
    top = (new_size[1] - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # 裁剪出中心区域
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

def get_video(video_path):
    import decord
    from decord import VideoReader
    from PIL import Image
    from mm_utils.constants import MAX_FRAMES, MIN_FRAMES
    import numpy as np

    video_reader = VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # dynamic sampling at 1fps, 1 frame per second between [MIN_FRAMES, MAX_FRAMES]
    frame_num = MAX_FRAMES
    frame_num = int(min(frame_num, duration))
    frame_num = int(max(frame_num, MIN_FRAMES))

    frame_indices = np.linspace(0, len(video_reader) - 1, frame_num, dtype=int)

    images_all = []
    video = video_reader.get_batch(frame_indices).asnumpy()
    for frame in video:
        images_all.append(Image.fromarray(frame))

    return images_all
    
def preprocess_qwen(sources: List[Dict[str, str]], tokenizer: "PreTrainedTokenizer", has_image: bool = True, max_len: int = 2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # start from human
    st = 0
    while sources[st]["from"] != "human":
        st += 1
    sources = sources[st:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [IGNORE_INDEX] * (len(system))
    assert len(input_id) == len(target)

    for j, sentence in enumerate(sources):
        role = roles[sentence["from"]]
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(DEFAULT_IMAGE_START_TOKEN).input_ids + [IMAGE_TOKEN_INDEX] + tokenizer(DEFAULT_IMAGE_END_TOKEN).input_ids + tokenizer(sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '')).input_ids + [im_end] + nl_tokens
        else:
            assert DEFAULT_IMAGE_TOKEN not in sentence["value"]
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id

        if role == "<|im_start|>user":
            _target = [IGNORE_INDEX] * (len(_input_id))
        elif role == "<|im_start|>assistant":
            _target = [IGNORE_INDEX] * len(tokenizer(role).input_ids) + [IGNORE_INDEX] + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + [IGNORE_INDEX]
        else:
            raise NotImplementedError
        target += _target

    assert len(input_id) == len(target)

    input_ids = torch.tensor(input_id, dtype=torch.long)
    targets = torch.tensor(target, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(seq_len)
        labels=targets,  # tensor(seq_len)
    )

from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors, cleanup_pcd, get_grid_size
from pcd.transform import Compose
def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]

    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))