"""
Collate functions for data loading within training and testing.
"""
from typing import Tuple
import torch

def mix_collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This feeds the model batches (assumes input 'batch' is batch_size audio files)
    The input 'batch' consists of a list of tuples (input, target, speaker_id, utterance_id),
    with batch_size elements.

    for each list element, the __getitem__ function of the dataset is called once.
    The inputs 'input' and 'target' should have shape (channels, time).
    """
    batch_size = len(batch)

    in_ch = batch[0][0].shape[0]
    tar_ch = batch[0][1].shape[0]
    remaining_shape = batch[0][0].shape[1:]
    dtype = batch[0][0].dtype

    all_inputs = torch.zeros(batch_size, in_ch, *remaining_shape, dtype=dtype)
    all_targets = torch.zeros(batch_size, tar_ch, *remaining_shape, dtype=dtype)

    for b in range(len(batch)):

        b_input, b_target = batch[b] # x, y; ignore speaker and utterance ID

        all_inputs[b, ...] = b_input
        all_targets[b, ...] = b_target

    # x, y
    if all_inputs.is_complex():
        return all_inputs, all_targets
    else:
        return all_inputs.float(), all_targets.float()

