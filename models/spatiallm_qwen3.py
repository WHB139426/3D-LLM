import torch
import torch.nn as nn
import os
import sys
import math
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3ForCausalLM, Qwen3Config
from transformers.modeling_utils import PreTrainedModel

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from models.sonata_encoder import Sonata, fourier_encode_vector

class PointProjector(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_in)
        self.linear_1 = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(dim_out, dim_out)

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class SpatialQwen3(PreTrainedModel):

    config_class = Qwen3Config

    def __init__(
        self,
        config= Qwen3Config.from_pretrained('/home/haibo/haibo_workspace/weights/Qwen3-0.6B'),
        sonata_path="/home/haibo/haibo_workspace/weights/sonata",
        llm_path='/home/haibo/haibo_workspace/weights/Qwen3-0.6B',
        tokenizer_model_max_length=32*1024,
        torch_dtype=torch.bfloat16,
        num_bins=1280,
    ):
        super().__init__(config)
        self.torch_dtype = torch_dtype
        self.point_hidden_size = 512

        # self.point_backbone = Sonata(
        #     in_channels=6,
        #     order=["z","z-trans","hilbert","hilbert-trans"],
        #     stride=[2,2,2,2],
        #     enc_depths=[3,3,3,12,3],
        #     enc_channels=[48,96,192,384,512],
        #     enc_num_head=[3,6,12,24,32],
        #     enc_patch_size=[1024,1024,1024,1024,1024],
        #     mlp_ratio=4,
        #     mask_token=True,
        #     enc_mode=True,
        #     enable_fourier_encode=True,
        #     num_bins=num_bins,
        # )
        # sonata_path = '/home/haibo/haibo_workspace/weights/sonata/spatiallm_qwen_0_5B_sonata.pth'
        # self.point_backbone.load_state_dict(torch.load(sonata_path, map_location='cpu'), strict=True)

        self.point_backbone = Sonata.from_pretrained("/home/haibo/haibo_workspace/weights/sonata")
        self.in_proj = nn.Linear(6, 9)
        self.coord_proj = nn.Linear(self.point_hidden_size + 63, self.point_hidden_size)

        self.language_model = Qwen3ForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch_dtype, tie_word_embeddings=False,
            attn_implementation="flash_attention_2" if torch_dtype==torch.bfloat16 else 'eager',
        )
        self.point_projector = PointProjector(self.point_hidden_size, self.language_model.config.hidden_size)

        self.config = self.language_model.config
        self.config.pad_token_id = 151643
        self.config.video_token_id = 151656
        self.tokenizer_model_max_length = tokenizer_model_max_length

        self.floating_point_ops = lambda s: 0

    def load_resume_ckpt(self, resume_path):
        from safetensors.torch import load_file
        loaded_state_dict = load_file(resume_path)
        self.load_state_dict(loaded_state_dict, strict=True)

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        labels, 
        input_pcd, 
    ):
        """
        input_ids: [bs, seq_len]
        labels: [bs, seq_len]
        input_pcd: [bs, point_num, 9]
        """
        if isinstance(input_ids, List):
            batch_size = len(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]
        # encode point clouds:
        point_features = []
        # self.point_backbone.to(torch.float32)
        for batch_idx in range(batch_size):
            point_cloud = input_pcd[batch_idx]
            nan_mask = torch.isnan(point_cloud).any(dim=1)
            point_cloud = point_cloud[~nan_mask]
            coords = point_cloud[:, :3].int()
            feats = point_cloud[:, 3:].float()
            with torch.cuda.amp.autocast(enabled=True, dtype=self.torch_dtype):
                # normals = torch.zeros_like(feats[:, :3]) # pad normal features with zeros
                # feats = torch.cat([feats, normals], dim=1)
                input_dict = {
                    "coord": feats[:, :3],
                    "grid_coord": coords,
                    "feat": self.in_proj(feats),
                    # "feat": feats,
                    "batch": torch.zeros(coords.shape[0], dtype=torch.long).to(point_cloud.device),
                }
                encoded_features, coords = self.point_backbone(input_dict)
                coords_normalised = coords / (self.point_backbone.reduced_grid_size - 1)
                encoded_coords = fourier_encode_vector(coords_normalised)
                encoded_features = torch.cat([encoded_features, encoded_coords], dim=-1)
                encoded_features = self.coord_proj(encoded_features)
                encoded_features = self.point_projector(encoded_features)
            point_features.append(encoded_features)

        new_input_ids = []
        new_labels = []
        new_input_embeds = []
        for batch_idx in range(batch_size):
            current_input_ids = input_ids[batch_idx] # [seq_len]
            if labels is not None:
                current_labels = labels[batch_idx] # [seq_len]
            current_point_embeds = point_features[batch_idx] # [point_len, dim]

            # scatter current_input_ids and current_labels: [seq_len+point_len-1]
            image_token_index_position = current_input_ids == IMAGE_TOKEN_INDEX
            current_input_ids = torch.cat([current_input_ids[:(i := (image_token_index_position).nonzero()[0].item())], current_input_ids.new_full((current_point_embeds.shape[0],), self.config.video_token_id), current_input_ids[i+1:]])
            if labels is not None:
                current_labels = torch.cat([current_labels[:(i := (image_token_index_position).nonzero()[0].item())], current_labels.new_full((current_point_embeds.shape[0],), IGNORE_INDEX), current_labels[i+1:]])

            # check video token number
            n_video_tokens = (current_input_ids == self.config.video_token_id).sum().item()
            if n_video_tokens != current_point_embeds.shape[0]:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {current_point_embeds.shape[0]}"
                )

            # scatter current_inputs_embeds with current_point_embeds
            current_inputs_embeds = self.language_model.get_input_embeddings()(current_input_ids) # [seq_len+point_len-1, dim]
            mask = current_input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(current_inputs_embeds)
            video_mask = mask_expanded.to(current_inputs_embeds.device)
            current_point_embeds = current_point_embeds.to(current_inputs_embeds.device, current_inputs_embeds.dtype)
            current_inputs_embeds = current_inputs_embeds.masked_scatter(video_mask, current_point_embeds) # [seq_len+point_len-1, dim]

            new_input_ids.append(current_input_ids)
            if labels is not None:
                new_labels.append(current_labels)
            new_input_embeds.append(current_inputs_embeds)

        # truncate tokenizer_model_max_length
        new_input_ids = [x[:self.tokenizer_model_max_length] for x in new_input_ids]
        new_input_embeds = [x[:self.tokenizer_model_max_length] for x in new_input_embeds]
        if labels is not None:
            new_labels = [x[:self.tokenizer_model_max_length] for x in new_labels]
        else:
            new_labels = [None for x in new_input_ids]

        # Combine them and padding
        max_len = max(x.shape[0] for x in new_input_embeds)
        new_input_embeds_padded = []
        new_input_ids_padded = torch.full((batch_size, max_len), self.config.pad_token_id, dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        if labels is not None:
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_input_ids_padded[0].device)

        for i, (cur_input_ids, cur_new_embed, cur_new_labels) in enumerate(zip(new_input_ids, new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_input_ids_padded[i, :cur_len] = cur_input_ids
                if labels is not None:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                pad_mask = (cur_input_ids == self.config.pad_token_id)
                attention_mask[i, :cur_len][pad_mask] = False

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) # [bs, seq_len+point_len-1, dim]

        if labels is not None:
            return None, None, attention_mask, new_input_embeds, new_labels_padded
        else:
            return None, None, attention_mask, new_input_embeds, None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_pcd: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        (input_ids, position_ids, attention_mask, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, labels, input_pcd)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if labels is not None:
            hidden_dim = hidden_states.size(-1)
            shift_labels = labels[..., 1:].contiguous().reshape(-1)
            shift_hidden_states = hidden_states[..., :-1, :].contiguous().reshape(-1, hidden_dim)
            assert shift_labels.size(0) == shift_hidden_states.size(0)
            mask = shift_labels > -1
            assert mask.float().sum() > 0
            shift_labels = shift_labels[mask]
            shift_hidden_states = shift_hidden_states[mask, :]
            logits = self.language_model.lm_head(shift_hidden_states)
            logits = logits.float()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, shift_labels)
        else:
            loss = None
            logits = self.language_model.lm_head(hidden_states)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# from transformers import InternVLForConditionalGeneration
# internvl = InternVLForConditionalGeneration.from_pretrained('/home/haibo/haibo_workspace/weights/InternVL3_5-1B-HF')
# qwen3 = Qwen3ForCausalLM.from_pretrained('/home/haibo/haibo_workspace/weights/Qwen3-0.6B')
# qwen3.model = internvl.model.language_model
# qwen3.lm_head = internvl.lm_head
# qwen3.save_pretrained('/home/haibo/haibo_workspace/weights/InternVL3_5-1B-HF/language_model')



# import os
# import sys
# import random
# sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
# from datasets.spatiallm import SpatialLMDataset
# from mm_utils.utils import *
# dataset = SpatialLMDataset()

# device = 'cuda:0'
# torch_dtype = torch.bfloat16
# model = SpatialQwen3(
#         config= Qwen3Config.from_pretrained('/home/haibo/haibo_workspace/weights/Qwen3-0.6B'),
#         sonata_path="/home/haibo/haibo_workspace/weights/sonata",
#         llm_path='/home/haibo/haibo_workspace/weights/Qwen3-0.6B',
#         tokenizer_model_max_length=32*1024,
#         torch_dtype=torch_dtype,
#         num_bins=1280,
#     ).to(device)
# print(get_parameter_number(model))
# input_ids = []
# labels = []
# input_pcd = []
# for i in range(333, 333+2):
#     item = random.choice(dataset)
#     # item = dataset[i]
#     print(item['scene_id'])
#     input_ids.append(item['input_ids'].to(device))
#     labels.append(item['labels'].to(device))
#     input_pcd.append(item['input_pcd'].to(device))

# print(input_ids[0].shape, input_ids[1].shape)
# print(labels[0].shape, labels[1].shape)
# print(input_pcd[0].shape, input_pcd[1].shape)

# with torch.inference_mode():
#     with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
#         outputs = model(input_ids=input_ids, labels=labels, input_pcd=input_pcd)
#     print(outputs.loss)
#     print(outputs.logits.shape)