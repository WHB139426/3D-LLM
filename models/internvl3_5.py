import torch
import torch.nn as nn
import os
import sys
import math
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import InternVLConfig, InternVLPreTrainedModel
from transformers.models.internvl.modeling_internvl import InternVLMultiModalProjector, InternVLVisionModel
from transformers import Qwen3ForCausalLM

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

class InternVLForConditionalGeneration(InternVLPreTrainedModel, GenerationMixin):

    config_class = InternVLConfig

    def __init__(
        self,
        config: InternVLConfig,
        tokenizer_model_max_length=32*1024,
    ):
        super().__init__(config)

        self.config = config
        self.config.pad_token_id = 151643
        self.config.video_token_id = 151656
        self.tokenizer_model_max_length = tokenizer_model_max_length

        self.vision_tower = InternVLVisionModel._from_config(config.vision_config)
        self.multi_modal_projector = InternVLMultiModalProjector(config)
        self.language_model = Qwen3ForCausalLM._from_config(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        self.floating_point_ops = lambda s: 0
        self.post_init()

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor of shape (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError("Height and width must be divisible by scale_factor for proper downsampling.")

        # Reshape to allow downsampling
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()
        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )
        # Swap height and width back for proper orientation
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        labels, 
        pixel_values_videos, 
    ):
        """
        input_ids: [bs, seq_len]
        labels: [bs, seq_len]
        pixel_values_videos: [bs, frame_num, 3, 384, 384]
        """
        if isinstance(input_ids, List):
            batch_size = len(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]

        # encode videos:
        video_features = []
        for batch_idx in range(batch_size):
            vision_features = self.vision_tower(pixel_values_videos[batch_idx]).last_hidden_state[:, 1:, :]  # [frame_num, 1024, 1024]
            # Calculate dimensions based on vision features for pixel shuffle
            channels = vision_features.shape[1]
            feature_size = int(channels**0.5)
            vision_features = vision_features.reshape(vision_features.shape[0], feature_size, feature_size, -1)
            vision_features = self.pixel_shuffle(vision_features, scale_factor=0.5)
            vision_features = vision_features.reshape(vision_features.shape[0], -1, vision_features.shape[-1]) # [frame_num, 256, 1024]
            vision_features = self.multi_modal_projector(vision_features) # [frame_num, 256, dim]
            """
            @FIX ME: The input should be 'Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}' in InternVL, 
            but we skip the text 'Frame_xx' for simplicity
            """
            vision_features = rearrange(vision_features, 'f n d -> (f n) d') # [frame_num*256, dim]
            video_features.append(vision_features)

        new_input_ids = []
        new_labels = []
        new_input_embeds = []

        for batch_idx in range(batch_size):
            current_input_ids = input_ids[batch_idx] # [seq_len]
            if labels is not None:
                current_labels = labels[batch_idx] # [seq_len]
            current_video_embeds = video_features[batch_idx] # [frame_num*256, dim]

            # scatter current_input_ids and current_labels: [seq_len+frame_num*256-1]
            image_token_index_position = current_input_ids == IMAGE_TOKEN_INDEX
            current_input_ids = torch.cat([current_input_ids[:(i := (image_token_index_position).nonzero()[0].item())], current_input_ids.new_full((current_video_embeds.shape[0],), self.config.video_token_id), current_input_ids[i+1:]])
            if labels is not None:
                current_labels = torch.cat([current_labels[:(i := (image_token_index_position).nonzero()[0].item())], current_labels.new_full((current_video_embeds.shape[0],), IGNORE_INDEX), current_labels[i+1:]])

            # check video token number
            n_video_tokens = (current_input_ids == self.config.video_token_id).sum().item()
            if n_video_tokens != current_video_embeds.shape[0]:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {current_video_embeds.shape[0]}"
                )

            # scatter current_inputs_embeds with current_video_embeds
            current_inputs_embeds = self.language_model.get_input_embeddings()(current_input_ids) # [seq_len+frame_num*256-1, dim]
            mask = current_input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(current_inputs_embeds)
            video_mask = mask_expanded.to(current_inputs_embeds.device)
            current_video_embeds = current_video_embeds.to(current_inputs_embeds.device, current_inputs_embeds.dtype)
            current_inputs_embeds = current_inputs_embeds.masked_scatter(video_mask, current_video_embeds) # [seq_len+frame_num*256-1, dim]

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

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) # [bs, seq_len+frame_num*256-1, dim]

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
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        (input_ids, position_ids, attention_mask, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, labels, pixel_values_videos)
 
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

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
# from datasets.video_r1 import VideoR1SFT
# dataset = VideoR1SFT(processor_path='/home/haibo/haibo_workspace/weights/InternVL3_5-1B-HF',)

# device = 'cuda:0'
# model = InternVLForConditionalGeneration.from_pretrained(
#     '/home/haibo/haibo_workspace/weights/InternVL3_5-1B-HF', 
#     tokenizer_model_max_length=32*1024,
#     attn_implementation="flash_attention_2", 
#     torch_dtype=torch.bfloat16,
#     ).to(device)

# input_ids = []
# labels = []
# pixel_values_videos = []
# for i in range(2):
#     item = dataset[i]
#     input_ids.append(item['input_ids'].to(device))
#     labels.append(item['labels'].to(device))
#     pixel_values_videos.append(item['pixel_values_videos'].to(device))

# print(input_ids[0].shape, input_ids[1].shape)
# print(labels[0].shape, labels[1].shape)
# print(pixel_values_videos[0].shape, pixel_values_videos[1].shape)

# with torch.inference_mode():
#     with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
#         outputs = model(input_ids=input_ids, labels=labels, pixel_values_videos=pixel_values_videos)
#         print(outputs.loss)
#         print(outputs.logits.shape)
