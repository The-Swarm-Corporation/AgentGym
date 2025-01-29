# Agent Gym
![Agent Gym](images/steps.png)


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

Convert any model into a r1-like reasoning hyper-intelligent agent. Leverages TRL, Huggingface, and various other libraries. This is a work in progress. Our goal is to make it easy to train any model into a reasoning agent.


- Sources:
- [Open R1 Blog](https://huggingface.co/blog/open-r1)
- [GRPO Documentation from trl](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Huggingface Docs](https://huggingface.co/docs/transformers/main/en/index)
- [GRPO Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)


## Installation

```bash
pip3 install -U agentgym
```

## Usage

```python
from agentgym.r1_pipeline import R1Pipeline, SFTConfig

r1_pipeline = R1Pipeline(sft_model="gpt2", sft_dataset="stanfordnlp/imdb", sft_args=SFTConfig(output_dir="/tmp"))

r1_pipeline.run()
```

## Architecture

The architecture is as follows:

- SFT: Supervised Fine-Tuning
- GRPO: Generative Reinforcement Policy Optimization

-> model -> sft -> grpo -> model

```mermaid
graph TD;
    A[model] --> B[sft]
    B --> C[grpo]
    C --> D[reasoning model]
```

# License
MIT
