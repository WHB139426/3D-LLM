import torch
import torch.nn as nn
import os
import sys
import math
import random
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3ForCausalLM, Qwen3Config
from transformers.modeling_utils import PreTrainedModel

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mm_utils.utils import *
from models.spatiallm_qwen3 import SpatialQwen3

MODEL_NAME_OR_PATH="/home/haibo/haibo_workspace/weights/Qwen3-0.6B"
POINT_DIR="/home/haibo/haibo_workspace/weights/sonata"
RESUME_PATH="/home/haibo/haibo_workspace/checkpoints/SpatialLM-Qwen3-0.6B-Further-FT-ScanRef-Multi3DRef/checkpoint-25000/model.safetensors"

device = 'cuda:3'
torch_dtype = torch.bfloat16
model = SpatialQwen3(
        config= Qwen3Config.from_pretrained(MODEL_NAME_OR_PATH),
        sonata_path=POINT_DIR,
        llm_path=MODEL_NAME_OR_PATH,
        tokenizer_model_max_length=32*1024,
        torch_dtype=torch_dtype,
        num_bins=1280,
)
print(get_parameter_number(model))

from safetensors.torch import load_file
loaded_state_dict = load_file(RESUME_PATH)
model.load_state_dict(loaded_state_dict, strict=True)
model.to(device)

# from datasets.spatiallm import SpatialLMDataset
# dataset = SpatialLMDataset(anno_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/spatiallm_val.json',)
# from datasets.scannet import ScannetDataset
# dataset = ScannetDataset(split_path = '/home/haibo/haibo_workspace/data/scannet-dataset/val.json',)
from datasets.scanref import ScanRefDataset
dataset = ScanRefDataset(split_path = '/home/haibo/haibo_workspace/data/scanref/ScanRefer_filtered_val.json',)
# from datasets.multi3dref import Multi3DRefDataset
# dataset = Multi3DRefDataset(split_path = '/home/haibo/haibo_workspace/data/multi3drefer_train_val/multi3drefer_val.json',)
# from datasets.referit3d import Refit3DDataset
# dataset = Refit3DDataset(split_path = '/home/haibo/haibo_workspace/data/referit3d/nr3d.csv',)

input_ids = []
labels = []
input_pcd = []
coord_min = []
grid_size = []
scene_ids = []
for i in range(1):
    item = random.choice(dataset)
    print(item['scene_id'])
    scene_ids.append(item['scene_id'])
    print('input_ids: ', dataset.tokenizer.decode(item['input_ids']))
    input_ids.append(item['input_ids'].to(device))
    labels.append(item['labels'].to(device))
    input_pcd.append(item['input_pcd'].to(device))
    coord_min.append(item['coord_min'])
    grid_size.append(item['grid_size'])

print(input_ids[0].shape)
print(labels[0].shape)
print(input_pcd[0].shape)
print(coord_min[0])
print(grid_size[0])

generate_kwargs = {
    "do_sample": False,
    "num_beams": 1, 
    "min_length": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 2048,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}
with torch.inference_mode():
    with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        (input_ids, position_ids, attention_mask, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(input_ids, labels, input_pcd)
        # Inference: Generation of the output
        generated_ids = model.language_model.generate(inputs_embeds=inputs_embeds, **generate_kwargs)
        output_text = dataset.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        print('\n\noutput_text: ', output_text)

        labels[0][labels[0] == -100] = 151643
        gt_text = dataset.tokenizer.decode(labels[0]).replace('<|endoftext|>','').replace('<|im_end|>','')
        print('\n\n\ngt_text: ', gt_text)

        output_and_gt_text = output_text.replace('Bbox(', 'Bbox(pred ') + '\n' + gt_text.replace('Bbox(', 'Bbox(gt ')


import re
import numpy as np

def denormalize_bbox(output_text, coord_min, grid_size):
    """
    将多行归一化的 Bbox 字符串还原为原始尺度（返回多行字符串）
    """
    pattern = (
        r"Bbox\(\s*([^,]+)\s*,\s*([\d\.-]+)\s*,\s*([\d\.-]+)\s*,\s*([\d\.-]+)\s*,"
        r"\s*([\d\.-]+)\s*,\s*([\d\.-]+)\s*,\s*([\d\.-]+)\s*,\s*([\d\.-]+)\s*\)"
    )

    coord_min = np.array(coord_min, dtype=float)
    grid_size = float(grid_size)
    results = []

    # 按行遍历每个 Bbox
    for idx, line in enumerate(output_text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if not match:
            raise ValueError(f"Invalid format in line {idx}: {line}")

        cls = match.group(1)
        nums = [float(match.group(i)) for i in range(2, 9)]
        x_n, y_n, z_n, angle_z, sx_n, sy_n, sz_n = nums

        x = x_n * grid_size + coord_min[0]
        y = y_n * grid_size + coord_min[1]
        z = z_n * grid_size + coord_min[2]
        sx = sx_n * grid_size
        sy = sy_n * grid_size
        sz = sz_n * grid_size

        result_str = f"bbox_{idx}=Bbox({cls},{x:.4f},{y:.4f},{z:.4f},{angle_z:.4f},{sx:.4f},{sy:.4f},{sz:.4f})"
        results.append(result_str)

    return "\n".join(results)


pred = denormalize_bbox(output_and_gt_text, coord_min[0], grid_size[0])

# import shutil
# import os
# with open('/home/haibo/haibo_workspace/data/SpatialLM-Dataset/examples/pred.txt', "w", encoding="utf-8") as f:
#     f.write(pred + "\n")

# src = f"/home/haibo/haibo_workspace/data/SpatialLM-Dataset/pcd/{scene_ids[0]}.ply"
# dst = "/home/haibo/haibo_workspace/data/SpatialLM-Dataset/examples/pred.ply"
# shutil.copy(src, dst)

# src = f"/home/haibo/haibo_workspace/data/SpatialLM-Dataset/layout/{scene_ids[0]}.txt"
# dst = "/home/haibo/haibo_workspace/data/SpatialLM-Dataset/examples/gt.txt"
# shutil.copy(src, dst)



import shutil
import os
with open('/home/haibo/haibo_workspace/data/pred.txt', "w", encoding="utf-8") as f:
    f.write(pred + "\n")

src = os.path.join(f'/home/haibo/haibo_workspace/data/scannet-dataset/{scene_ids[0]}/', scene_ids[0]+'_vh_clean_2.ply')
dst = "/home/haibo/haibo_workspace/data/pred.ply"
shutil.copy(src, dst)