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
    point_name_or_path: Optional[str] = field(default="Sonata")
    resume_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    num_bins: Optional[int] = field(default=1280)
    dataset: Optional[str] = field(default="spatiallm")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    mpt_attn_impl: Optional[str] = field(default="triton")
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    model_max_length: int = field(default=32*1024, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    point_lr: Optional[float] = None

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
            processor_path=processor_path,
            num_bins = data_args.num_bins,
        )
        train_dataset.append(scannet_dataset)
        print('add scannet')
    if 'scanref' in data_args.dataset:
        from datasets.scanref import ScanRefDataset
        scanref_dataset = ScanRefDataset(
            split_path = '/home/haibo/haibo_workspace/data/scanref/ScanRefer_filtered_train.json',
            anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
        )
        train_dataset.append(scanref_dataset)
        print('add scanref')
    if 'multi3dref' in data_args.dataset:
        from datasets.multi3dref import Multi3DRefDataset
        multi3dref_dataset = Multi3DRefDataset(
            split_path = '/home/haibo/haibo_workspace/data/multi3drefer_train_val/multi3drefer_train.json',
            anno_path = '/home/haibo/haibo_workspace/data/scannet-dataset',
            processor_path=processor_path,
            num_bins = data_args.num_bins,
        )
        train_dataset.append(multi3dref_dataset)
        print('add multi3dref')

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

    # enable lora
    if training_args.lora_enable:
        raise NotImplementedError

    # tokenizer to save
    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=False)

    # data_module and trainer
    data_module = make_supervised_data_module(data_args=data_args, processor_path=model_args.model_name_or_path)
    trainer = SFT_Trainer(model=model, args=training_args, processing_class=tokenizer, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()