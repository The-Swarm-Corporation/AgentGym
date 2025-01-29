# sft -> grpo -> sft + grpo -> new model


import uuid
from typing import Callable, List, Optional

from datasets import load_dataset
from loguru import logger
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)


def generate_model_uuid():
    return str(uuid.uuid4())


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
        *args,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.sft_dataset = load_dataset(sft_dataset, split="train")
        self.sft_model = AutoModelForCausalLM.from_pretrained(
            sft_model
        )
        self.saved_model_file_path = saved_model_file_path
        self.multi_gpu = multi_gpu
        self.peft_config = peft_config
        self.reward_funcs = reward_funcs
        self.sft_lora_only = sft_lora_only
        self.liger_kernel_on = liger_kernel_on
        self.model_name = model_name
        self.saved_model_file_path = f"{self.output_dir}/{model_name}_{generate_model_uuid()}.pth"

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
            self.save_model_weights()
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
