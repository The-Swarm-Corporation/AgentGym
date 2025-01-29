# !pip install agentgym

from agentgym.r1_pipeline import R1Pipeline, SFTConfig

r1_pipeline = R1Pipeline(
    sft_model="Qwen/Qwen2-0.5B-Instruct",
    tokenizer_name="Qwen/Qwen2-0.5B-Instruct",
    sft_dataset="trl-lib/tldr",
    sft_args=SFTConfig(output_dir="/tmp"),
    only_grpo=True,
    model_name="Qwen/Qwen2-0.5B-Instruct"
)

r1_pipeline.run()
