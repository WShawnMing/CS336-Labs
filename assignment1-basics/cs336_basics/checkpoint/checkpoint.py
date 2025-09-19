import torch


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save the model and optimizer state to a checkpoint file.
    """
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration
        }, out)


def load_checkpoint(src, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]