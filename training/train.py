import torch
import tokenizers
import transformers
import pathlib
from packaging import version
import yaml
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from training.trainer_sft import SFT_Trainer

"""
if 'GLIBCXX_3.4.32 not found'
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/workspace/miniconda3/envs/<the specific conda env>/lib/
or
    conda install -c conda-forge libstdcxx-ng
"""
local_rank = None
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-0.6B")
    unfreeze_point_backbone: bool = field(default=False)
    unfreeze_vision_tower: bool = field(default=False)
    point_name_or_path: Optional[str] = field(default="Sonata")
    resume_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    num_bins: Optional[int] = field(default=1280)
    num_frames: Optional[int] = field(default=32)
    dataset: Optional[str] = field(default="scanref")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    mpt_attn_impl: Optional[str] = field(default="triton")
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    model_max_length: int = field(default=32*1024, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    vision_tower_lr: Optional[float] = None
    point_lr: Optional[float] = None

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_tower"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    lora_module_names = list(lora_module_names)

    return lora_module_names

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    torch.cuda.synchronize()
    trainer.save_model(output_dir)
    return

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        batch["input_ids"] = [instance["input_ids"] for instance in instances]
        batch["labels"] = [instance["labels"] for instance in instances]
        if "pixel_values_videos" in instances[0].keys():
                batch["pixel_values_videos"] = [instance["pixel_values_videos"] for instance in instances]
        if "input_pcd" in instances[0].keys():
                batch["input_pcd"] = [instance["input_pcd"] for instance in instances]
        return batch

def make_supervised_data_module(data_args, processor_path) -> Dict[str, Any]:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = []
    if 'spatiallm' in data_args.dataset:
        from datasets.spatiallm import SpatialLMDataset
        spatiallm_dataset = SpatialLMDataset(
            anno_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/spatiallm_train.json',
            pcd_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/pcd',
            layout_path = '/home/haibo/haibo_workspace/data/SpatialLM-Dataset/layout',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
        )
        train_dataset.append(spatiallm_dataset)
        print('add spatiallm')
    if 'scannet' in data_args.dataset:
        from datasets.scannet import ScannetDataset
        scannet_dataset = ScannetDataset(
            split_path = '/home/haibo/haibo_workspace/data/scannet-dataset/train.json',
            anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
            video_path = '/home/haibo/haibo_workspace/data/scannet-frames',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
            num_frames = data_args.num_frames,
        )
        train_dataset.append(scannet_dataset)
        print('add scannet')
    if 'scanref' in data_args.dataset:
        from datasets.scanref import ScanRefDataset
        scanref_dataset = ScanRefDataset(
            split_path = '/home/haibo/haibo_workspace/data/scanref/ScanRefer_filtered_train.json',
            anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
            video_path = '/home/haibo/haibo_workspace/data/scannet-frames',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
            num_frames = data_args.num_frames,
        )
        train_dataset.append(scanref_dataset)
        print('add scanref')
    if 'multi3dref' in data_args.dataset:
        from datasets.multi3dref import Multi3DRefDataset
        multi3dref_dataset = Multi3DRefDataset(
            split_path = '/home/haibo/haibo_workspace/data/multi3drefer_train_val/multi3drefer_train.json',
            anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
            video_path = '/home/haibo/haibo_workspace/data/scannet-frames',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
            num_frames = data_args.num_frames,
        )
        train_dataset.append(multi3dref_dataset)
        print('add multi3dref')
    if 'referit3d' in data_args.dataset:
        from datasets.referit3d import Refit3DDataset
        referit3d_dataset = Refit3DDataset(
            split_path = '/home/haibo/haibo_workspace/data/referit3d/nr3d.csv',
            anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
            video_path = '/home/haibo/haibo_workspace/data/scannet-frames',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
            num_frames = data_args.num_frames,
        )
        train_dataset.append(referit3d_dataset)
        print('add referit3d')
    data_collator = DataCollatorForSupervisedDataset()

    if len(train_dataset) == 1:
        train_dataset = train_dataset[0]
    else:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_dataset)

    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    if 'Qwen3' in model_args.model_name_or_path:
        """
        point inputs llm
        """
        from models.spatiallm_qwen3 import SpatialQwen3
        from transformers import Qwen3Config
        model = SpatialQwen3(
            config=Qwen3Config.from_pretrained(model_args.model_name_or_path),
            sonata_path=model_args.point_name_or_path,
            llm_path=model_args.model_name_or_path,
            tokenizer_model_max_length=training_args.model_max_length,
            torch_dtype=compute_dtype,
            num_bins=data_args.num_bins,
        )
        if model_args.resume_path != None:
            model.load_resume_ckpt(model_args.resume_path)
        # freeze point_backbone while unfreeze projector
        if not model_args.unfreeze_point_backbone:
            for name, param in model.point_backbone.named_parameters():
                param.requires_grad = False
    elif 'InternVL3_5' in model_args.model_name_or_path:
        """
        video inputs llm
        """
        from models.internvl3_5 import InternVLForConditionalGeneration
        model = InternVLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, 
                                        tokenizer_model_max_length=training_args.model_max_length,
                                        attn_implementation="flash_attention_2", torch_dtype=compute_dtype,)
        model.config.use_cache = False
        # freeze vision_tower while unfreeze projector
        if not model_args.unfreeze_vision_tower:
            for name, param in model.vision_tower.named_parameters():
                param.requires_grad = False

    # enable lora
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # processor to save
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=False)

    # data_module and trainer
    data_module = make_supervised_data_module(data_args=data_args, processor_path=model_args.model_name_or_path)
    trainer = SFT_Trainer(model=model, args=training_args, processing_class=processor, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()