import os
import sys
import random
import traceback
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor, LlavaOnevisionProcessor, InternVLProcessor
from qwen_vl_utils import fetch_video
import decord
from decord import VideoReader

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.constants import DEFAULT_IMAGE_TOKEN, MAX_PIXELS, MIN_PIXELS, MIN_FRAMES, MAX_FRAMES
from mm_utils.utils import load_json, preprocess_qwen, resize_and_center_crop

class VideoR1SFT(Dataset):
    def __init__(
        self,
        anno_path = '/home/haibo/haibo_workspace/data/Video-R1-data/Video-R1-COT-165k.json',
        video_path = '/home/haibo/haibo_workspace/data/Video-R1-data',
        processor_path='/home/haibo/haibo_workspace/weights/Qwen2.5-VL-7B-Instruct/',
    ):
        super().__init__()
        raw_annos = load_json(anno_path)
        self.annos = [item for item in raw_annos if item['data_type']=='video' and 'ego4d' not in item['path']]
        self.video_path = video_path

        if 'Qwen2.5-VL' in processor_path:
            self.processor_type = 'qwen2.5_vl'
            self.processor = Qwen2_5_VLProcessor.from_pretrained(processor_path, use_fast=False)
        elif 'llava-onevision' in processor_path:
            self.processor_type = 'llava_ov'
            self.processor = LlavaOnevisionProcessor.from_pretrained(processor_path, use_fast=False)
        elif 'InternVL3_5' in processor_path:
            self.processor_type = 'internvl_3.5'
            self.processor = InternVLProcessor.from_pretrained(processor_path, use_fast=False)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        """Handle exceptions since this dataset includes some bad files."""
        try:
            return self.get_item(i)
        except Exception:
            traceback.print_exc()
            backup_idx = random.randint(0, len(self) - 1)
            print(self.annos[i]['problem_id'], self.annos[i]['path'])
            print(f"Encounted error when process {i}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)

    def get_qwen_vl_video(self, data_info):
        video_path = os.path.join(self.video_path, data_info['path'].replace('./', ''))
        video = fetch_video({
            'type': 'video', 
            'video': video_path, 
            'max_pixels': MAX_PIXELS, 'min_pixels': MIN_PIXELS, 'min_frames': MIN_FRAMES, 'max_frames': MAX_FRAMES, 'fps': 1,
            })
        return video

    def get_llava_ov_video(self, data_info):
        video_path = os.path.join(self.video_path, data_info['path'].replace('./', ''))
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

    def get_item(self, i):
        data_info = self.annos[i]
        question = data_info['problem']
        options = data_info['options']

        # prepare instruction input
        instruction = 'Question: ' + question
        if len(options) != 0:
            instruction += '\nOptions:'
            for i in range(len(options)):
                instruction += "\n" + options[i]

        # prepare response output
        think_process = data_info['process']
        answer = data_info['solution']
        response = think_process + answer
        # prepare input_ids and labels
        conversations = [
            {'from': 'human', 'value': DEFAULT_IMAGE_TOKEN + instruction},
            {'from': 'gpt', 'value': response},
            ]
        data_dict = preprocess_qwen(conversations, self.processor.tokenizer)

        # prepare video inputs
        if self.processor_type == 'qwen2.5_vl':
            video = self.get_qwen_vl_video(data_info)
            inputs = self.processor(
                    text=['<|vision_start|><|video_pad|><|vision_end|>'],
                    images=None,
                    videos=video,
                    fps=1,
                    padding=True,
                    return_tensors="pt",
                )
            data_dict['pixel_values_videos'] = inputs['pixel_values_videos']
            data_dict['question_ids'] = data_info['problem_id']
            data_dict['video_ids'] = data_info['path']
            data_dict['video_grid_thw'] = inputs['video_grid_thw']
            data_dict['second_per_grid_ts'] = inputs['second_per_grid_ts']

        elif self.processor_type == 'llava_ov':
            video = self.get_llava_ov_video(data_info)
            pixel_values_videos = []
            for frame in video:
                frame = resize_and_center_crop(frame, 384)
                pixel_values = self.processor.image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0][0] # [3, 384, 384]
                pixel_values_videos.append(pixel_values)
            data_dict['pixel_values_videos'] = torch.stack(pixel_values_videos, dim=0) # [T, 3, 384, 384]
            data_dict['question_ids'] = data_info['problem_id']
            data_dict['video_ids'] = data_info['path']
            
        elif self.processor_type == 'internvl_3.5':
            video = self.get_llava_ov_video(data_info)
            pixel_values_videos = []
            for frame in video:
                frame = resize_and_center_crop(frame, 448)
                pixel_values = self.processor.image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0] # [3, 448, 448]
                pixel_values_videos.append(pixel_values)
            data_dict['pixel_values_videos'] = torch.stack(pixel_values_videos, dim=0) # [T, 3, 448, 448]
            data_dict['question_ids'] = data_info['problem_id']
            data_dict['video_ids'] = data_info['path']

        return data_dict

# dataset = VideoR1SFT(processor_path='/home/haibo/haibo_workspace/weights/InternVL3_5-8B-HF',)
# for i in range(1):
#     item = random.choice(dataset)
#     print(item['video_ids'], item['question_ids'])
#     print(item['input_ids'].shape, item['labels'].shape)

#     # print(item['input_ids'])
#     # print(item['labels'])
#     # item['input_ids'][item['input_ids'] == 151656] = 151647

#     print('input_ids: ', dataset.processor.tokenizer.decode(item['input_ids']))
#     item['labels'][item['labels'] == -100] = 151643
#     print('labels: ', dataset.processor.tokenizer.decode(item['labels']))
#     print(item['pixel_values_videos'].shape)
#     print()
# print(len(dataset))