import torch


class LASLossWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, log_probs : torch.Tensor, text_encoded: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        return {"loss": torch.nn.CrossEntropyLoss()(log_probs.reshape(-1, log_probs.size(-1)), text_encoded.to(torch.long).view(-1))}
