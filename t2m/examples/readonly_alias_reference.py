"""Actual overlapping source spans and an independent ordered-dot reference."""

import numpy as np

from .strided_memory_reference import arguments, output_arguments


def aliased_arguments(m,n,k,strides,dtype,*,writable=False,offset=11):
    values,_ = (output_arguments(m,n,k,strides,(-47,2),dtype) if writable
                else arguments(m,n,k,strides,dtype))
    begin = 5
    owner_size = max(begin+len(values[0]),offset+len(values[1]))+7
    numbers = ((np.arange(owner_size,dtype=np.int32)%31)-15).astype(np.float32)/np.float32(8)
    owner = ((numbers.view(np.uint32)>>16).astype(np.uint16) if dtype == "bf16"
             else numbers.astype(np.float16 if dtype == "f16" else np.float32))
    values[0],values[1] = owner[begin:begin+len(values[0])],owner[offset:offset+len(values[1])]

    def view(storage,rows,cols,row,col,origin):
        if not rows or not cols:
            return np.empty((rows,cols),np.float32)
        observed = np.ndarray((rows,cols),dtype=storage.dtype,buffer=storage,
                              offset=int(origin)*storage.itemsize,strides=(row*storage.itemsize,col*storage.itemsize))
        return (observed.astype(np.uint32)<<16).view(np.float32) if dtype == "bf16" else observed.astype(np.float32)

    origin = 11 if writable else 10
    left = view(values[0],m,k,*strides[:2],values[origin])
    right = view(values[1],k,n,*strides[2:],values[origin+1])
    dense = values[2].copy()
    for inner in range(((k+31)//32)*32):
        product = left[:,inner:inner+1]*right[inner:inner+1,:] if inner < k else np.zeros((m,n),np.float32)
        dense = np.add(dense,product,dtype=np.float32)
    if writable:
        expected = values[3].copy()
        if m and n:
            target = np.ndarray((m,n),dtype=np.float32,buffer=expected,offset=int(values[17])*4,
                                strides=(int(values[15])*4,int(values[16])*4))
            target[...] = dense
    else:
        expected = dense
    return values,expected,owner
