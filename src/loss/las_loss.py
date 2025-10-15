import torch


class LASLossWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, log_probs : torch.Tensor, text_encoded: torch.Tensor, text_encoded_length: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        loss = torch.tensor(0.0, device=log_probs.device)

        for log_prob, encoded, lens in zip(log_probs, text_encoded, text_encoded_length):
            selected_log_probs = log_prob[: lens, encoded[: lens].to(torch.long)]
            loss += -selected_log_probs.mean()

        return {"loss": loss / len(log_probs)}
