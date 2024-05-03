import functools, operator 

import torch
import torch.nn.functional as F
import warnings

from .task_configs import task_parameters as TASK_PARAMETERS

DEFAULT_MASK_EXTRA_RADIUS=5

def make_mask(tensor, task):
    return build_mask(tensor.unsqueeze(1), val=task_parameters[task]['mask_val'])[0]

def build_mask(target, val=0.0, tol=1e-3, mask_extra_radius=DEFAULT_MASK_EXTRA_RADIUS):
    if target.shape[1] == 1:
        mask = ((target >= val - tol) & (target <= val + tol))
        #mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
        mask = (0 != F.conv2d(mask.float(),
                        torch.ones(1, 1, mask_extra_radius, mask_extra_radius, device=mask.device),
                        padding=(mask_extra_radius // 2)) )
        return (~mask).expand_as(target)


    masks = [target[:, t] for t in range(target.shape[1])]
    masks = [(t >= val - tol) & (t <= val + tol) for t in masks]
    mask = functools.reduce(lambda a,b: a&b, masks).unsqueeze(1)
    mask = (0 != F.conv2d(mask.float(),
                        torch.ones(1, 1, mask_extra_radius, mask_extra_radius, device=mask.device),
                        padding=(mask_extra_radius // 2)) )
#     mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
    return (~mask).expand_as(target)


def make_mask_from_data(tensors, tasks, mask_extra_radius=DEFAULT_MASK_EXTRA_RADIUS, device='cpu'):
    ''' Makes mask from a list of tensors (and their associated tasks)
        Args:
            tensors: Iterable of torch.tensors
            tasks: List of tasks, named as in .task_configs.py
        Returns:
            Mask to be used for all the image tasks in `tensors`
    '''
    task_to_tensor = {task: tensor for task, tensor in zip(tasks, tensors)}
    if 'mask_valid' in tasks: # resizing might mess with exact values, so prioritize mask_val.
        return build_mask(task_to_tensor['mask_valid'],
                          mask_extra_radius=mask_extra_radius,
                          val=TASK_PARAMETERS['mask_valid']['mask_val']) 

    # could be improved by priotizing images w/fewer channels
    for tensor, task in zip(tensors, tasks):
        if 'mask_val' in TASK_PARAMETERS[task]:
            tensor_size = tensor.shape[-1]
            if tensor_size != 512 and task != 'mask_valid':
                warnings.warn(f"Making mask from task \"{task}\" with size {tensor_size}, which has been resized from 512. Masks may be inaccurate.", RuntimeWarning)
            return build_mask(tensor,
                              mask_extra_radius=mask_extra_radius,
                              val=TASK_PARAMETERS[task]['mask_val'])

    raise ValueError(f'Could not make mask for any task in {tasks}')
    