"""Readonly snapshots preserve words without hiding writable overlap."""

import numpy as np
import pytest

from t2m.effect_reference import validate_inputs
from t2m.input_storage import MAX_ALIAS_WORK
from t2m.shared_tensor import TensorABI, TensorBuffer
from t2m.tensor_storage import DTYPES


def spec(dtype="f32", *, writable=False):
    return TensorABI("f",(TensorBuffer("x",(8,),dtype=dtype),TensorBuffer("y",(8,),dtype=dtype),
                           TensorBuffer("state",(8,),"readwrite" if writable else "write",dtype)))


@pytest.mark.parametrize("dtype",["f16","bf16","f32","f64","i64","i1","f8E4M3FNUZ"])
@pytest.mark.parametrize("identical",[False,True])
def test_readonly_overlap_snapshots_preserve_logical_words(dtype,identical):
    dt = np.dtype(DTYPES[dtype])
    raw = np.arange(16*dt.itemsize,dtype=np.uint8)
    if dtype == "i1": raw %= 2
    owner = raw.view(dt)
    if dtype in ("f16","bf16","f32","f64","f8E4M3FNUZ"):
        words = {
            "f16":[0x7E11,0xFE22,0,0x8000,0x3C00],
            "bf16":[0x7FC1,0xFFC2,0,0x8000,0x3F80],
            "f32":[0x7FC01234,0xFFC05678,0,0x80000000,0x3F800000],
            "f64":[0x7FF8000000001234,0xFFF8000000005678,0,0x8000000000000000,0x3FF0000000000000],
            "f8E4M3FNUZ":[0x80,0x7F,0,0x38,0xFE],
        }[dtype]
        owner = np.resize(np.asarray(words,dtype=f"uint{dt.itemsize*8}"),16).view(dt)
    arrays = (owner[:8],owner[:8] if identical else owner[4:12])
    before = [value.tobytes() for value in arrays]
    assert np.shares_memory(*arrays)
    prepared = validate_inputs(spec(dtype),arrays)
    assert not np.shares_memory(*prepared)
    for i,value in enumerate(prepared):
        assert value.dtype == dt and value.tobytes() == before[i]
        assert arrays[i].tobytes() == before[i]
        assert not np.shares_memory(value,owner)


def test_complete_overlap_graph_is_checked_before_any_snapshot():
    owner = np.arange(16,dtype=np.float32)
    arrays = (owner[:8],owner[4:12],owner[8:16])
    before = owner.tobytes()
    with pytest.raises(ValueError,match="writable source storage.*y, state"):
        validate_inputs(spec(writable=True),arrays)
    assert owner.tobytes() == before and np.shares_memory(arrays[0],arrays[1])


def test_interleaved_disjoint_writable_values_do_not_inherit_overlap():
    owner = np.arange(16,dtype=np.float32)
    arrays = (np.arange(8,dtype=np.float32),owner[::2],owner[1::2])
    assert np.may_share_memory(arrays[1],arrays[2])
    assert not np.shares_memory(arrays[1],arrays[2])
    prepared = validate_inputs(spec(writable=True),arrays)
    assert all(a is b for a,b in zip(arrays,prepared,strict=True))


def test_uncertain_writable_overlap_refuses_without_copying(monkeypatch):
    owner = np.arange(16,dtype=np.float32)
    arrays = (np.arange(8,dtype=np.float32),owner[::2],owner[1::2])
    called = []
    def unavailable(left,right,*,max_work):
        called.append(max_work)
        raise np.exceptions.TooHardError("bounded fixture")
    monkeypatch.setattr(np,"shares_memory",unavailable)
    with pytest.raises(ValueError,match="bounded analysis"):
        validate_inputs(spec(writable=True),arrays)
    assert called == [MAX_ALIAS_WORK]


def test_readonly_possible_overlap_needs_no_expensive_exact_search(monkeypatch):
    owner = np.arange(16,dtype=np.float32)
    def forbidden(*args,**kwargs):
        raise AssertionError("readonly values need only conservative snapshots")
    monkeypatch.setattr(np,"shares_memory",forbidden)
    arrays = (owner[::2],owner[1::2])
    prepared = validate_inputs(spec(),arrays)
    assert all(a.tobytes() == b.tobytes() for a,b in zip(arrays,prepared,strict=True))
    assert all(value.flags.owndata for value in prepared)


@pytest.mark.parametrize("effect",["read","readwrite"])
def test_scalar_carriers_are_snapshotted_by_value_without_copying_tensor_storage(effect):
    from t2m.scalar_abi import RuntimeScalar
    buffers = (TensorBuffer("tensor",(8,),effect),TensorBuffer("result",(8,),"write")) if effect == "read" else (TensorBuffer("tensor",(8,),effect),)
    abi = TensorABI("f",buffers,runtime_scalars=(RuntimeScalar("n","i32",len(buffers),bounds=(0,9)),))
    owner = np.arange(8,dtype=np.int32)
    tensor,scalar = owner.view(np.float32),owner[:1].reshape(())
    prepared = validate_inputs(abi,[tensor,scalar])
    assert prepared[0] is tensor and prepared[1].item() == 0
    assert not np.shares_memory(prepared[1],tensor)
    owner[0] = 7
    assert prepared[1].item() == 0 and scalar.item() == 7
