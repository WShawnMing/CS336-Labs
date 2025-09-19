import torch
import numpy as np
import numpy.typing as npt


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """
    assert len(dataset) >= context_length + batch_size, "Dataset length must be greater than or equal to context length + batch size"
    max_start_index = len(dataset) - context_length - 1 # 确定最大起始索引
    start_indices = np.random.randint(0, max_start_index + 1, size=batch_size) # 随机生成起始索引
    
    # 直接在 numpy 中构建批次，然后一次性转换为 tensor
    inputs_list = [dataset[s : s + context_length] for s in start_indices]
    targets_list = [dataset[s + 1 : s + context_length + 1] for s in start_indices]
    
    inputs = torch.tensor(np.array(inputs_list), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)

    return inputs, targets


def __main__():
    dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batch_size = 2
    context_length = 3
    device = "cpu"
    dataset, labels = get_batch(dataset, batch_size, context_length, device)
    print(dataset)
    print(labels)

# if __name__ == "__main__":
#     __main__()