# sft -> grpo -> sft + grpo -> new model


import subprocess
import sys
import uuid
from typing import Callable, List, Optional

import torch
from accelerate import PartialState
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from agentgym.reward_funcs import (
    correctness_reward_func,
    format_reward_func,
    int_reward_func,
    reward_func_for_format,
    reward_len,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)



class GRPOArgs(BaseModel):
    output_dir: Optional[str] = Field(None, alias="output_dir")
    run_name: Optional[str] = Field(None, alias="run_name")
    learning_rate: Optional[float] = Field(5e-6, alias="learning_rate")
    adam_beta1: Optional[float] = Field(0.9, alias="adam_beta1")
    adam_beta2: Optional[float] = Field(0.99, alias="adam_beta2")
    weight_decay: Optional[float] = Field(0.1, alias="weight_decay")
    warmup_ratio: Optional[float] = Field(0.1, alias="warmup_ratio")
    lr_scheduler_type: Optional[str] = Field('cosine', alias="lr_scheduler_type")
    logging_steps: Optional[int] = Field(1, alias="logging_steps")
    bf16: Optional[bool] = Field(True, alias="bf16")
    per_device_train_batch_size: Optional[int] = Field(1, alias="per_device_train_batch_size")
    gradient_accumulation_steps: Optional[int] = Field(4, alias="gradient_accumulation_steps")
    num_generations: Optional[int] = Field(16, alias="num_generations")
    max_prompt_length: Optional[int] = Field(256, alias="max_prompt_length")
    max_completion_length: Optional[int] = Field(786, alias="max_completion_length")
    num_train_epochs: Optional[int] = Field(1, alias="num_train_epochs")
    save_steps: Optional[int] = Field(100, alias="save_steps")
    max_grad_norm: Optional[float] = Field(0.1, alias="max_grad_norm")
    report_to: Optional[str] = Field("wandb", alias="report_to")
    log_on_each_node: Optional[bool] = Field(False, alias="log_on_each_node")

prebuilt_reward_funcs = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    reward_len,
    reward_func_for_format,
    format_reward_func,
]

def generate_model_uuid():
    """
    This function generates a short UUID.
    """
    return str(uuid.uuid4())[:8]

def check_gpu_availability():
    """
    This function checks if a GPU is available.
    """
    if torch.cuda.is_available():
        return True
    else:
        return False


class R1Pipeline:
    def __init__(
        self,
        output_dir: str = "/tmp",
        sft_dataset: str = "stanfordnlp/imdb",
        sft_model: str = "facebook/opt-350m",
        sft_args: SFTConfig = SFTConfig(output_dir="/tmp"),
        saved_model_file_path: str = None,
        reward_funcs: List[Callable] = [],
        multi_gpu: bool = False,
        sft_lora_only: bool = False,
        liger_kernel_on: bool = False,
        peft_config: Optional[LoraConfig] = peft_config,
        model_name: str = "agent-gym-r1",
        check_gpu_availability: bool = check_gpu_availability,
        grpo_args: GRPOArgs = GRPOArgs(),
        tokenizer_name: str = "None",
        use_prebuilt_reward_funcs: bool = True,
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.sft_args = sft_args
        self.sft_dataset = load_dataset(sft_dataset, split="train")
        device_string = PartialState().process_index
        
        self.sft_model = AutoModelForCausalLM.from_pretrained(
            sft_model,
            device_map = {'':device_string} if multi_gpu else None
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.saved_model_file_path = saved_model_file_path
        self.multi_gpu = multi_gpu
        self.peft_config = peft_config
        self.reward_funcs = reward_funcs
        self.sft_lora_only = sft_lora_only
        self.liger_kernel_on = liger_kernel_on
        self.model_name = model_name
        self.check_gpu_availability = check_gpu_availability
        self.grpo_args = grpo_args
        self.use_prebuilt_reward_funcs = use_prebuilt_reward_funcs
        self.saved_model_file_path = f"{self.output_dir}/{model_name}_{generate_model_uuid()}.pth"
        
        self.check_for_flash_attention()

        self.sft_trainer = SFTTrainer(
            model=self.sft_model,
            train_dataset=self.sft_dataset,
            args=self.sft_args,
            peft_config=(
                self.peft_config if sft_lora_only is True else None
            ),
            use_liger=(
                self.liger_kernel_on
                if liger_kernel_on is True
                else False
            ),
            *args,
            **kwargs,
        )

    def sft_train(self, *args, **kwargs):
        # run the training loop
        try:
            logger.info("Starting training...")
            self.sft_trainer.train(*args, **kwargs)
            logger.info("Training completed successfully")
            return self.save_model_weights()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise e

    def save_model_weights(self):
        try:
            logger.info("Saving model weights...")
            self.sft_trainer.save_model(self.saved_model_file_path)
            logger.info("Model weights saved successfully")
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            raise e
        
    def check_for_flash_attention(self):
        try:
            device = check_gpu_availability()
            
            if device is True:
                subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn"], check=True)
                self.sft_model.attn_implementation="flash_attention_2",
                logger.info("Flash attention 2 is enabled")
            else:
                logger.info("Flash attention 2 is not enabled")
        except Exception as e:
            logger.error(f"Error checking for flash attention: {e}")
            raise e


    def load_grpo_args(self):
        config = GRPOArgs(**self.grpo_args)
        training_args = GRPOConfig(**config)
        return training_args
    
    
    def grpo_train(self, model_path: str, *args, **kwargs):
        try:
            logger.info("Starting GRPO training...")
            training_args = self.load_grpo_args()
            reward_funcs = prebuilt_reward_funcs if self.use_prebuilt_reward_funcs else self.reward_funcs
            
            trainer = GRPOTrainer(
                model=model_path,
                processing_class=self.tokenizer,
                reward_funcs=reward_funcs,
                args=training_args,
                train_dataset=self.sft_dataset,
                *args,
                **kwargs,
            )
            
            trainer.train()
            
            trainer.save_model(self.saved_model_file_path)
            
            logger.info(f"GRPO training completed successfully and model saved to: {self.saved_model_file_path}")
            
            return self.saved_model_file_path
        except Exception as e:
            logger.error(f"Error during GRPO training: {e}")
            raise e
        
        
    def run(self):
        try:
            logger.info("Starting R1 pipeline with SFT first and then GRPO")
            model = self.sft_train()
            logger.info(f"SFT training completed successfully and model saved to: {model}")
            model = self.grpo_train(model)
            logger.info(f"GRPO training completed successfully and model saved to: {model}")
            logger.info("R1 pipeline completed successfully")
            return model
        except Exception as e:
            logger.error(f"Error during R1 pipeline: {e}")
            raise e
