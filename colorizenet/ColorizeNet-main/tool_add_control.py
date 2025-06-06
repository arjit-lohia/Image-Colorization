# import sys
# import os

# assert len(sys.argv) == 3, f'Number of args ({len(sys.argv)}) must be 3.'

# input_path = sys.argv[1]
# output_path = sys.argv[2]

# assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

# import torch
# # from share import *
# from cldm.model import create_model


# def get_node_name(name, parent_name):
#     if len(name) <= len(parent_name):
#         return False, ''
#     p = name[:len(parent_name)]
#     if p != parent_name:
#         return False, ''
#     return True, name[len(parent_name):]


# model = create_model(config_path='./models/cldm_v21.yaml')

# pretrained_weights = torch.load(input_path)
# if 'state_dict' in pretrained_weights:
#     pretrained_weights = pretrained_weights['state_dict']

# scratch_dict = model.state_dict()

# target_dict = {}
# for k in scratch_dict.keys():
#     is_control, name = get_node_name(k, 'control_')
#     if is_control:
#         copy_k = 'model.diffusion_' + name
#     else:
#         copy_k = k
#     if copy_k in pretrained_weights:
#         target_dict[k] = pretrained_weights[copy_k].clone()
#     else:
#         target_dict[k] = scratch_dict[k].clone()
#         print(f'These weights are newly added: {k}')

# model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
# print('Done.')


import sys
import os
import torch
# from share import *
from cldm.model import create_model
import pytorch_lightning.callbacks.model_checkpoint as pl_checkpoint

# Register ModelCheckpoint as a safe global (Fix for PyTorch 2.6+)
torch.serialization.add_safe_globals([pl_checkpoint.ModelCheckpoint])

assert len(sys.argv) == 3, f'Number of args ({len(sys.argv)}) must be 3.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

model = create_model(config_path='./models/cldm_v21.yaml')

# ✅ Fix: Explicitly allow full loading to prevent UnpicklingError
pretrained_weights = torch.load(input_path, weights_only=False)

if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
