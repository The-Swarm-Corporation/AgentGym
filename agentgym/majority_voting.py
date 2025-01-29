
import torch
from litellm import encode


class BaseModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MajorityVoting(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = []

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_accuracy(self, answer: str, target: str) -> float:
        # first convert to tensors and then compute cosine similarity
        answer_tokens = encode(model="gpt-4o", text=answer)
        target_tokens = encode(model="gpt-4o", text=target)

        answer_tensor = torch.tensor(
            answer_tokens, dtype=torch.float32
        )
        target_tensor = torch.tensor(
            target_tokens, dtype=torch.float32
        )

        if answer_tensor.dim() == 1:
            answer_tensor = answer_tensor.unsqueeze(0)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)

        return (
            torch.cosine_similarity(
                answer_tensor, target_tensor, dim=1
            )
            .mean()
            .item()
        )


vote = MajorityVoting()

print(vote.compute_accuracy("hello", "chicken"))
