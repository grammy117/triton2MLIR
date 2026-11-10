"""Independent source-allocation state for aliased C/output GEMM pointers."""

import numpy as np

from .strided_memory_reference import output_arguments


def arguments(m,n,k,strides,out_strides,dtype,*,c_origin=5,sensitive=False):
    values,output = output_arguments(m,n,k,strides,out_strides,dtype,sensitive=sensitive)
    count = max(len(values[3]),c_origin+m*n+13) if m and n else len(values[3])
    words = np.asarray([0x7FC01234,0xFFC05678,0,0x80000000,0x3F800000,0xBF800000],np.uint32)
    state = np.resize(words,count).view(np.float32)
    if m and n:
        state[c_origin:c_origin+m*n] = values[2].ravel()
    expected = state.copy()
    if m and n:
        view = np.ndarray((m,n),dtype=np.float32,buffer=expected,offset=int(values[17])*4,
                          strides=(int(values[15])*4,int(values[16])*4))
        reference = np.ndarray((m,n),dtype=np.float32,buffer=output,offset=int(values[17])*4,
                               strides=(int(values[15])*4,int(values[16])*4))
        view[...] = reference
    values[18] = np.asarray(count,np.int32)
    return [*values[:2],state,*values[4:],np.asarray(c_origin,np.int64)],expected
