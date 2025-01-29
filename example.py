from agentgym.r1_pipeline import R1Pipeline, SFTConfig

r1_pipeline = R1Pipeline(
    sft_model="gpt2",
    tokenizer_name="gpt2",
    sft_dataset="stanfordnlp/imdb",
    sft_args=SFTConfig(output_dir="/tmp"),
    only_grpo=True,
    model_name="gpt-2"
)

r1_pipeline.run()
