import os
import sys
import re
from torch.utils.data import Dataset
from transformers import AutoProcessor

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors, cleanup_pcd, get_grid_size
from pcd.transform import Compose
from mm_utils.constants import DEFAULT_IMAGE_TOKEN, DEC_PROMPT

def load_bboxes_from_txt(file_path: str):
    labels = []
    bboxes = []
    pattern = re.compile(
        r"Bbox\(([^,]+),([-\d\.eE]+),([-\d\.eE]+),([-\d\.eE]+),([-\d\.eE]+),([-\d\.eE]+),([-\d\.eE]+),([-\d\.eE]+)\)"
    )
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("bbox_"):
                match = pattern.search(line)
                if match:
                    label = match.group(1).strip()
                    values = list(map(float, match.groups()[1:]))
                    labels.append(label)
                    bboxes.append(values)
    return labels, bboxes  # labels: list[str], bboxes: (N,7)

class SpatialLMDataset(Dataset):
    def __init__(
        self,
        anno_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/spatiallm_train.json',
        pcd_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/pcd',
        layout_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/layout',
        processor_path='/home/haibo/haibo_workspace/weights/Qwen3-0.6B',
        num_bins = 1280,
    ):
        super().__init__()
        raw_annos = load_json(anno_path)
        self.annos = [item['point_clouds'][0].replace('.ply', '').replace('pcd/', '') for item in raw_annos]
        self.pcd_path = pcd_path
        self.layout_path = layout_path
        self.num_bins = num_bins
        self.tokenizer = AutoProcessor.from_pretrained(processor_path, use_fast=False)

    def __len__(self):
        return len(self.annos)
    
    def __getitem__(self, i):
        scene_id = self.annos[i]

        # extract layout
        labels, bboxes = load_bboxes_from_txt(os.path.join(self.layout_path, scene_id+'.txt'))
        assert len(labels) == len(bboxes)

        # load point cloud
        point_cloud = load_o3d_pcd(os.path.join(self.pcd_path, scene_id+'.ply'))
        grid_size = get_grid_size(self.num_bins)
        point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)
        points, colors = get_points_and_colors(point_cloud)
        coord_min = np.min(points, 0) # [x_min, y_min, z_min] for PositiveShift
        input_pcd = preprocess_point_cloud(points, colors, grid_size, self.num_bins)[0] # [N, 9] 9: coord, xyz, rgb

        # prepare prompt and response
        prompt = DEC_PROMPT
        response_lines = []
        for label, bbox in zip(labels, bboxes):
            x, y, z, angle_z, sx, sy, sz = bbox
            x = int((x-coord_min[0]) / grid_size)
            y = int((y-coord_min[1]) / grid_size)
            z = int((z-coord_min[2]) / grid_size)
            angle_z = angle_z
            sx = int(sx / grid_size)
            sy = int(sy / grid_size)
            sz = int(sz / grid_size)
            response_lines.append(
                f"Bbox({label}, {x}, {y}, {z}, {angle_z:.4f}, {sx}, {sy}, {sz})"
            )
        response = "\n".join(response_lines)
        conversations = [
            {'from': 'human', 'value': DEFAULT_IMAGE_TOKEN + prompt},
            {'from': 'gpt', 'value': response},
            ]

        data_dict = preprocess_qwen(conversations, self.tokenizer)
        data_dict['scene_id'] = scene_id
        data_dict['input_pcd'] = input_pcd
        data_dict['coord_min'] = coord_min
        data_dict['grid_size'] = grid_size

        return data_dict


        
# dataset = SpatialLMDataset()
# for i in range(10):
#     item = random.choice(dataset)
#     print(item['scene_id'])
#     print(item['input_ids'].shape, item['labels'].shape)

#     item['labels'][item['labels'] == -100] = 151643
#     print('input_ids: ', dataset.tokenizer.decode(item['input_ids']))
#     print('labels: ', dataset.tokenizer.decode(item['labels']))

#     print(item['input_pcd'].shape)
#     print()
# print(len(dataset))


