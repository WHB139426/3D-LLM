import numpy as np
import os
import sys
import re
import glob
from torch.utils.data import Dataset
from transformers import AutoProcessor
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors, cleanup_pcd, get_grid_size
from pcd.transform import Compose
from mm_utils.constants import DEFAULT_IMAGE_TOKEN, SCANREF_PROMPT

def read_axis_alignment(txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
    start = content.find('axisAlignment =') + len('axisAlignment =')
    end = content.find('colorHeight', start)
    axis_data_str = content[start:end].strip()
    nums = list(map(float, axis_data_str.split()))
    axis_alignment = np.array(nums).reshape(4, 4)
    return axis_alignment

def align_axis(points, axisAlignment):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_aligned = (axisAlignment @ points_h.T).T[:, :3]
    return points_aligned

def load_bboxes(ply_path, segs_path, agg_path, axisAlignment):
    ply = load_o3d_pcd(ply_path)
    segs_data = load_json(segs_path)
    agg_data = load_json(agg_path)

    all_colors = np.asarray(ply.colors)
    all_points = np.asarray(ply.points)
    segs_indices = np.array(segs_data["segIndices"])
    seg_groups = agg_data["segGroups"]

    all_bboxes = []
    all_labels = []

    for group in seg_groups:
        object_id = group.get("objectId")
        label = group.get("label", "unknown") # 获取标签，如果没有则为 unknown
        segments = set(group["segments"])

        point_mask = np.isin(segs_indices, list(segments))
        object_points = all_points[point_mask]
        object_points = align_axis(object_points, axisAlignment)

        min_bound = object_points.min(axis=0)
        max_bound = object_points.max(axis=0)
        bbox_center = (min_bound + max_bound) / 2.0
        bbox_size = max_bound - min_bound
        
        # [cx, cy, cz, angle_z, dx, dy, dz]
        bbox_7_digits = np.concatenate([bbox_center, [0], bbox_size])
        all_bboxes.append(bbox_7_digits)
        all_labels.append(label)
    
    return all_labels, np.array(all_bboxes)


class ScanRefDataset(Dataset):
    def __init__(
        self,
        split_path = '/home/haibo/haibo_workspace/data/scanref/ScanRefer_filtered_train.json',
        anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
        processor_path='/home/haibo/haibo_workspace/weights/InternVL3_5-1B-HF',
        video_path = '/home/haibo/haibo_workspace/data/scannet-frames',
        num_bins = 1280,
        num_frames = 32,
    ):
        super().__init__()
        raw_annos = load_json(split_path) 
        self.annos = [item for item in raw_annos if item['scene_id']!='scene0533_00'] # 'scene0533_00' missing in the scannet-frames
        self.anno_path = anno_path
        self.video_path = video_path
        self.num_bins = num_bins
        self.num_frames = num_frames

        self.processor = AutoProcessor.from_pretrained(processor_path, use_fast=False)
        self.tokenizer = self.processor.tokenizer

    def __len__(self):
        return len(self.annos)

    def get_frames(self, video_path):
        search_pattern = os.path.join(video_path, '*.jpg')
        all_jpg_files = sorted(glob.glob(search_pattern))

        if self.num_frames >= len(all_jpg_files):
            indices = np.arange(len(all_jpg_files))
        else:
            indices_float = np.linspace(0, len(all_jpg_files) - 1, num=self.num_frames)
            indices = np.round(indices_float).astype(int)

        selected_files = [all_jpg_files[i] for i in sorted(list(set(indices)))]
        images_all = []
        for file in selected_files:
            images_all.append(Image.open(file))

        return images_all

    def __getitem__(self, i):
        scene_id = self.annos[i]["scene_id"]
        object_id = int(self.annos[i]["object_id"])
        object_name = self.annos[i]["object_name"]
        description = self.annos[i]["description"].strip()

        ply_path = os.path.join(self.anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.ply')
        segs_path = os.path.join(self.anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.0.010000.segs.json')
        agg_path = os.path.join(self.anno_path+f'/{scene_id}', scene_id+'.aggregation.json')

        # axis alignment matrix
        axisAlignment = read_axis_alignment(os.path.join(self.anno_path+f'/{scene_id}', scene_id+'.txt'))

        # extract layout
        labels, bboxes = load_bboxes(ply_path, segs_path, agg_path, axisAlignment)
        assert len(labels) == len(bboxes)
        labels = [labels[object_id]]
        bboxes = [bboxes[object_id]]

        # load point cloud
        point_cloud = load_o3d_pcd(ply_path)
        # axis alignment
        points = np.asarray(point_cloud.points)
        points_aligned = align_axis(points, axisAlignment)
        point_cloud.points = o3d.utility.Vector3dVector(points_aligned)
        axis_aligned_point_cloud = point_cloud

        grid_size = get_grid_size(self.num_bins)
        point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)
        points, colors = get_points_and_colors(point_cloud)
        coord_min = np.min(points, 0) # [x_min, y_min, z_min] for PositiveShift
        input_pcd = preprocess_point_cloud(points, colors, grid_size, self.num_bins)[0] # [N, 9] 9: coord, xyz, rgb
        
        # prepare prompt and response
        prompt = SCANREF_PROMPT
        prompt = prompt.replace('<object_name>', labels[0]).replace('<object_description>', description)
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

        # get frames
        video_path = os.path.join(self.video_path, scene_id)
        video = self.get_frames(video_path)
        pixel_values_videos = []
        for frame in video:
            frame = resize_and_center_crop(frame, 448)
            pixel_values = self.processor.image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0] # [3, 448, 448]
            pixel_values_videos.append(pixel_values)
        data_dict['pixel_values_videos'] = torch.stack(pixel_values_videos, dim=0) # [T, 3, 448, 448]
        data_dict['scene_id'] = scene_id
        data_dict['input_pcd'] = input_pcd
        data_dict['coord_min'] = coord_min
        data_dict['grid_size'] = grid_size
        data_dict['object_name'] = object_name
        data_dict['description'] = description
        data_dict['axisAlignment'] = axisAlignment
        data_dict['axis_aligned_point_cloud'] = axis_aligned_point_cloud

        return data_dict

# dataset = ScanRefDataset()
# for i in range(10):
#     item = random.choice(dataset)
#     print(item['scene_id'])
#     print(item['input_ids'].shape, item['labels'].shape)

#     item['labels'][item['labels'] == -100] = 151643
#     print('input_ids: ', dataset.tokenizer.decode(item['input_ids']))
#     print('labels: ', dataset.tokenizer.decode(item['labels']).replace('<|endoftext|>',''))

#     print(item['pixel_values_videos'].shape, item['input_pcd'].shape)
#     print()
# print(len(dataset))