import torch
import torch.nn as nn
from tqdm import tqdm
import math
import re
import numpy as np
import torch.multiprocessing as mp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from models.spatiallm_qwen3 import SpatialQwen3

# cp /home/haibo/haibo_workspace/data/scannet-dataset/scene0663_00/scene0663_00_vh_clean_2.ply /home/haibo/haibo_workspace/data/pred.ply

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

    return [x, y, z, angle_z, sx, sy, sz]

def calculate_3d_iou_axis_aligned(gt_coord, pred_coord):
    """
    计算两个轴对齐的3D边界框的IoU (angle_z = 0)。
    
    坐标格式: [x, y, z, angle_z, sx, sy, sz] (angle_z被忽略)
    
    参数:
        gt_coord (list or np.array): 真实框的7个参数。
        pred_coord (list or np.array): 预测框的7个参数。
        
    返回:
        3D IoU值 (float)。
    """
    # 1. 解析输入参数，忽略角度
    gt_x, gt_y, gt_z, _, gt_sx, gt_sy, gt_sz = gt_coord
    pred_x, pred_y, pred_z, _, pred_sx, pred_sy, pred_sz = pred_coord

    # 2. 计算两个框在每个轴上的最小和最大坐标
    gt_min = np.array([gt_x - gt_sx / 2, gt_y - gt_sy / 2, gt_z - gt_sz / 2])
    gt_max = np.array([gt_x + gt_sx / 2, gt_y + gt_sy / 2, gt_z + gt_sz / 2])
    
    pred_min = np.array([pred_x - pred_sx / 2, pred_y - pred_sy / 2, pred_z - pred_sz / 2])
    pred_max = np.array([pred_x + pred_sx / 2, pred_y + pred_sy / 2, pred_z + pred_sz / 2])

    # 3. 计算交集区域的最小和最大坐标
    inter_min = np.maximum(gt_min, pred_min)
    inter_max = np.minimum(gt_max, pred_max)

    # 4. 计算交集区域的长、宽、高
    # 如果 inter_max 在某个维度上小于 inter_min，说明没有重叠，差值为负。
    # 我们用 np.maximum(0, ...) 来处理这种情况，确保非重叠维度长度为0。
    inter_dims = np.maximum(0, inter_max - inter_min)
    
    # 5. 计算交集体积
    intersection_volume = inter_dims[0] * inter_dims[1] * inter_dims[2]

    # 如果交集体积为0，则IoU也为0
    if intersection_volume == 0:
        return 0.0

    # 6. 计算各自的体积
    gt_volume = gt_sx * gt_sy * gt_sz
    pred_volume = pred_sx * pred_sy * pred_sz
    
    # 7. 计算并集体积
    union_volume = gt_volume + pred_volume - intersection_volume
    
    # 防止除以零
    if union_volume == 0:
        return 0.0

    # 8. 计算最终的3D IoU
    iou_3d = intersection_volume / union_volume
    
    return iou_3d

# 评测函数
def evaluate(rank, dataset, split_index_list):
    torch.cuda.set_device(rank)  # 绑定到特定GPU
    device = f'cuda:{rank}'


    torch_dtype = torch.bfloat16
    MODEL_NAME_OR_PATH="/home/haibo/haibo_workspace/weights/Qwen3-0.6B"
    POINT_DIR="/home/haibo/haibo_workspace/weights/sonata"
    CKPT="/home/haibo/haibo_workspace/checkpoints/SpatialLM-Qwen3-0.6B-Further-FT-ScanRef/model.safetensors"

    """
    initialize the model
    """
    from transformers import Qwen3Config
    model = SpatialQwen3(
            config= Qwen3Config.from_pretrained(MODEL_NAME_OR_PATH),
            sonata_path=POINT_DIR,
            llm_path=MODEL_NAME_OR_PATH,
            tokenizer_model_max_length=32*1024,
            torch_dtype=torch_dtype,
            num_bins=1280,
    )
    print(get_parameter_number(model))

    """
    load checkpoint
    """
    from safetensors.torch import load_file
    loaded_state_dict = load_file(CKPT)
    model.load_state_dict(loaded_state_dict, strict=True)
    model.to(device)

    """
    inference
    """
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

    result = []

    for index in tqdm(split_index_list, desc=f"GPU {rank}"):
        item = dataset[index]
        input_ids = []
        labels = []
        input_pcd = []

        input_ids.append(item['input_ids'].to(device))
        labels.append(item['labels'].to(device))
        input_pcd.append(item['input_pcd'].to(device))

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                (input_ids, position_ids, attention_mask, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(input_ids, labels, input_pcd)
                # Inference: Generation of the output
                generated_ids = model.language_model.generate(inputs_embeds=inputs_embeds, **generate_kwargs)
                output_text = dataset.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                labels[0][labels[0] == -100] = 151643
                gt_text = dataset.tokenizer.decode(labels[0]).replace('<|endoftext|>','').replace('<|im_end|>','')

                try:
                    pred_coord = denormalize_bbox(output_text, item['coord_min'], item['grid_size'])
                    gt_coord = denormalize_bbox(gt_text, item['coord_min'], item['grid_size'])
                except Exception:
                    print("denormalize_bbox Error: ", output_text)
                    continue
                result.append({
                    'scene_ids': item['scene_id'],
                    'input_text': dataset.tokenizer.decode(item['input_ids']),
                    'object_name': item['object_name'],
                    'description': item['description'],
                    'pred_text': output_text,
                    'gt_text': gt_text,
                    'pred_coord': pred_coord,
                    'gt_coord': gt_coord,
                    'box3d_iou': calculate_3d_iou_axis_aligned(pred_coord, gt_coord),
                })

        save_json(result, f'experiments/scanref_gpu{rank}.json')


# 启动多进程
if __name__ == "__main__":
    """
    load the dataset
    """
    from datasets.scanref import ScanRefDataset
    dataset = ScanRefDataset(split_path = '/home/haibo/haibo_workspace/data/scanref/ScanRefer_filtered_val.json',)

    import random
    index_list = [i for i in range(len(dataset))]
    random.shuffle(index_list)

    # 按GPU数目划分数据
    num_gpus = 4
    splits = np.array_split(index_list, num_gpus)

    mp.set_start_method('spawn')  # 确保能正确启动子进程
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=evaluate, args=(rank, dataset, splits[rank].tolist()))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并 JSON 结果
    final_res = []
    for rank in range(num_gpus):
        file_path = f'experiments/scanref_gpu{rank}.json'
        res_part = load_json(file_path)
        final_res.extend(res_part)
        os.remove(file_path)  # 删除子 JSON 文件

    save_json(final_res, 'experiments/scanref.json')