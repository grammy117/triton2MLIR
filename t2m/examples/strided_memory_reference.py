"""Independent NumPy source-allocation views and ordered tiled-dot reference.

The input archives contain complete owning allocations. Byte strides and an
origin construct actual signed/padded/broadcast views for the oracle, rather
than relabeling dense matrices as strided memory. Compiler code never imports
this reference.
"""

import numpy as np


def allocation(rows, cols, row_stride, col_stride, dtype, *, prefix=5, suffix=7):
    carrier = np.uint16 if dtype == "bf16" else np.float16 if dtype == "f16" else np.float32
    if not rows or not cols:
        return np.empty(0,carrier),np.empty((rows,cols),np.float32),0
    low = min(0,(rows-1)*row_stride)+min(0,(cols-1)*col_stride)
    high = max(0,(rows-1)*row_stride)+max(0,(cols-1)*col_stride)
    origin = prefix-low
    storage = np.full(prefix+high-low+suffix+1,0x7FC1 if dtype == "bf16" else np.nan,carrier)
    addresses = {origin+i*row_stride+j*col_stride for i in range(rows) for j in range(cols)}
    for address in addresses:
        value = np.float32((address*7)%31-15)/np.float32(8)
        storage[address] = np.asarray(value,np.float32).view(np.uint32).item() >> 16 if dtype == "bf16" else value
    view = np.ndarray((rows,cols),dtype=carrier,buffer=storage,offset=origin*storage.itemsize,
                      strides=(row_stride*storage.itemsize,col_stride*storage.itemsize))
    values = (view.astype(np.uint32)<<16).view(np.float32) if dtype == "bf16" else view.astype(np.float32)
    return storage,values,origin


def arguments(m,n,k,strides,dtype,*,sensitive=False):
    ar,ak,bk,bc = strides
    a,left,abase = allocation(m,k,ar,ak,dtype)
    b,right,bbase = allocation(k,n,bk,bc,dtype,prefix=3,suffix=9)
    initial = np.resize(np.asarray([0.,-0.,2.**24,-2.**24,2.,-3.],np.float32),(m,n))
    if sensitive:
        if dtype == "bf16":
            a.fill(0x3F80);b.fill(0x3F80)
        else:
            a.fill(1.);b.fill(1.)
        left.fill(1.);right.fill(1.);initial.fill(2.**24)
    output = initial.copy()
    # Match the original padded BK loop, including its zero products. The
    # small dyadic inputs have exact products in f32; additions round at each K.
    for inner in range(((k+31)//32)*32):
        if inner < k:
            product = left[:,inner:inner+1]*right[inner:inner+1,:]
        else:
            product = np.zeros((m,n),np.float32)
        output = np.add(output,product,dtype=np.float32)
    values = [a,b,initial,*(np.asarray(value,np.int32) for value in (m,n,k)),
              *(np.asarray(value,np.int64) for value in (*strides,abase,bbase)),
              np.asarray(len(a),np.int32),np.asarray(len(b),np.int32)]
    return values,output


def output_arguments(m,n,k,strides,out_strides,dtype,*,sensitive=False,conflicts=False):
    values,dense = arguments(m,n,k,strides,dtype,sensitive=sensitive)
    row,col = out_strides
    if m and n:
        low = min(0,(m-1)*row)+min(0,(n-1)*col)
        high = max(0,(m-1)*row)+max(0,(n-1)*col)
        origin,count = 7-low,7+high-low+13+1
        addresses = {origin+i*row+j*col for i in range(m) for j in range(n)}
        if len(addresses) != m*n and not conflicts:
            raise ValueError("reference requires independent logical output addresses")
    else:
        origin,count = 0,11
    # Keep noncanonical NaNs, signed zero and ordinary finite words in holes.
    words = np.asarray([0x7FC01234,0xFFC05678,0,0x80000000,0x3F800000,0xBF800000],np.uint32)
    initial = np.resize(words,count).view(np.float32)
    expected = initial.copy()
    if m and n and not conflicts:
        view = np.ndarray((m,n),dtype=np.float32,buffer=expected,offset=origin*4,strides=(row*4,col*4))
        view[...] = dense
    return [*values[:3],initial,*values[3:],*(np.asarray(v,np.int64) for v in (*out_strides,origin)),
            np.asarray(count,np.int32)],expected


def policy(*,words=False):
    return {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
            "results":{"archive":"reference.npz","comparisons":[{"kind":"words" if words else "ieee"}]}}
