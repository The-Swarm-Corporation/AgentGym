# Agent Gym


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

Agent Gym is a framework for training and evaluating reinforcement learning agents in a gym-like environment.

- Sources:
- [Open R1 Blog](https://huggingface.co/blog/open-r1)
- [GRPO Documentation from trl](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Huggingface Docs](https://huggingface.co/docs/transformers/main/en/index)

## Usage

```python
from agentgym.r1_pipeline import R1Pipeline, SFTConfig

r1_pipeline = R1Pipeline(sft_model="gpt2", sft_dataset="stanfordnlp/imdb", sft_args=SFTConfig(output_dir="/tmp"))

r1_pipeline.run()
```

# License
MIT
