import torch




class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.reshape(inputs, (-1 , inputs.shape[-1]))
        batch_size = inputs.shape[0]
        Max = torch.max(inputs, dim=1, keepdim=True)[0] # [batch_size, 1]
        log_sum = torch.log(torch.sum(torch.exp(inputs - Max), dim=1, keepdim=True)) # [batch_size, 1]
        # target_logits = inputs.gather(1, targets.unsqueeze(1))
        target_logits = inputs[torch.arange(batch_size), targets].unsqueeze(1)
        loss = log_sum + Max - target_logits # [batch_size, 1]
        loss = torch.sum(loss) / batch_size
        return loss






