import numpy as np
import torch
import math

## pack/unpackbits for torch tensors, 
# sourced from (https://gist.github.com/vadimkantorov/30ea6d278bc492abf6ad328c6965613a)
############################################################################################################

def _tensor_dim_slice(tensor, dim, dim_slice):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (dim_slice, )]

#@torch.jit.script
def _packshape(shape, dim : int = -1, mask : int = 0b00000001, dtype = torch.uint8, pack = True):
    dim = dim if dim >= 0 else dim + len(shape)
    bits = (8 if dtype is torch.uint8 else 
            16 if dtype is torch.int16 else 
            32 if dtype is torch.int32 else 
            64 if dtype is torch.int64 else 
            0)
    nibble = (1 if mask == 0b00000001 else 
              2 if mask == 0b00000011 else 
              4 if mask == 0b00001111 else 
              8 if mask == 0b11111111 else 
              0)
    # bits = torch.iinfo(dtype).bits # does not JIT compile
    assert nibble <= bits and bits % nibble == 0
    nibbles = bits // nibble
    if pack:
        shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)), ) + shape[1 + dim:])
    else:
        shape = (shape[:dim] + (shape[dim] * nibbles, ) + shape[1 + dim:])
    return shape, nibbles, nibble

#@torch.jit.script
def _packbits(tensor, dim : int = -1, mask : int = 0b00000001, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape, nibbles, nibble = _packshape(tensor.shape, dim = dim, mask = mask, dtype = dtype, pack = True)
    out = out if out is not None else torch.empty(shape, device = tensor.device, dtype = dtype)
    assert out.shape == shape
    
    if tensor.shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype = torch.uint8, device = tensor.device)
        shift = shift.view(nibbles, *((1, ) * (tensor.dim() - dim - 1)))
        torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift, dim = 1 + dim, out = out)
        
    else:
        for i in range(nibbles):
            shift = nibble * i
            sliced_input = _tensor_dim_slice(tensor, dim, slice(i, None, nibbles))
            sliced_output = out.narrow(dim, 0, sliced_input.shape[dim])
            if shift == 0:
                sliced_output.copy_(sliced_input)
            else:
                sliced_output.bitwise_or_(sliced_input << shift)
    return out

#@torch.jit.script
def _unpackbits(tensor, dim : int = -1, mask : int = 0b00000001, shape = None, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = _packshape(tensor.shape, dim = dim, mask = mask, dtype = tensor.dtype, pack = False)
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device = tensor.device, dtype = dtype)
    assert out.shape == shape
    
    if shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype = torch.uint8, device = tensor.device)
        shift = shift.view(nibbles, *((1, ) * (tensor.dim() - dim - 1)))
        return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out = out)
    
    else:
        for i in range(nibbles):
            shift = nibble * i
            sliced_output = _tensor_dim_slice(out, dim, slice(i, None, nibbles))
            sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
            torch.bitwise_and(sliced_input >> shift, mask, out = sliced_output)
    return out    

############################################################################################################


def binarize_descriptor(desc, axis=1):
    '''
    binarize and pack float32 descriptor to uint8 descriptor.

    the dimension of descriptors are reduced by 8 times.

    Parameters
    ----------
    desc: ndarray, float32 or float64 type, with shape (B,256,M) or (256,M)
    axis: int, the axis to pack, default 1

    Returns
    -------
    desc_binary: ndarray, uint8 type, with shape (B,32,M/8) or (32,M/8)
    '''

    assert desc.dtype == np.float32 or desc.dtype == np.float64
    assert desc.ndim == 2 or desc.ndim == 3
    assert axis < desc.ndim and axis >= -desc.ndim

    desc_binary = np.zeros(desc.shape)
    desc_binary[desc > 0] = 1
    desc_binary = desc_binary.astype(np.uint8)
    desc_binary = np.packbits(desc_binary, axis=axis)
    return desc_binary



def binarize_descriptor_tensor(desc, axis=1):
    '''
    binarize and pack float32 descriptor tensor to uint8 descriptor tensor.

    the dimension of descriptors are reduced by 8 times.

    Parameters
    ----------
    desc: tensor, float32 or float64 type, with shape (B,256,M) or (256,M)
    axis: int, the axis to pack, default 1

    Returns
    -------
    desc_binary: tensor, uint8 type, with shape (B,32,M/8) or (32,M/8)
    '''
    desc_binary = desc.clone()
    desc_binary[desc_binary > 0] = 1
    desc_binary = desc_binary.type(torch.uint8)
    desc_binary = _packbits(desc_binary, dim=axis)
    return desc_binary



def get_dist_matrix_binary(desc0, desc1):
    '''
    get hamming distance matrix between desc0 and desc1
    
    Parameters
    ----------
    desc0: ndarray, uint8 type, with shape (B,32,M)
    desc1: ndarray, uint8 type, with shape (B,32,N)
    
    Returns
    -------
    unpacked : ndarray, float32 type
        Hamming distance matrix with shape (B,M,N)
    '''
    desc0 = np.unpackbits(desc0, axis=-2) # (B,32,M) -> (B,256,M)
    desc1 = np.unpackbits(desc1, axis=-2) # (B,32,N) -> (B,256,N)

    hamming_distance = np.sum(desc0[:, :, :, None] != desc1[:, :, None, :], axis=1)
    hamming_distance = hamming_distance.astype(np.float32) / 256.
    return hamming_distance


    