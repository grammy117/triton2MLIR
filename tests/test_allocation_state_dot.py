"""Source C and stores share state while original tiled K reuse remains."""

import os

import pytest

from t2m.allocation_state_demo import convert
from t2m.examples.allocation_state_reference import arguments
from t2m.examples.strided_memory_reference import policy

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing shared/native/Merlin environments required")


@pytest.mark.parametrize("dtype",["f32","f16","bf16"])
def test_tiled_contraction_reads_original_C_from_shared_allocation_and_returns_all_words(tmp_path,dtype):
    from test_allocation_bindings import replay
    from test_mixed_effects import qualify
    result = convert(tmp_path / "source",dtype=dtype,optimizer=OPT)
    values,expected = arguments(7,11,37,(79,2,64,3),(-29,2),dtype)
    _,qualified = qualify(result,tmp_path,values,policy(words=True),[expected])
    assert len(result.spec.outputs) == 1 and len(result.spec.buffers) == 3
    contractions = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(contractions) == 1 and contractions[0].results[0].type.get_shape() == (16,16)
    assert "scf.for" in result.mlir_text and 'source accumulator/output pointers share' in result.mlir_text
    for index,(m,n,k,out_strides,origin,sensitive) in enumerate([
        (17,19,37,(19,1),7,False),  # Exact C/output aliases across programs.
        (5,7,65,(1,9),5,False),  # Partial overlap, one source tile.
        (3,5,0,(7,1),5,False),
        (7,11,37,(-29,2),5,True),
        (0,7,65,(-29,2),8191,False),
    ]):
        values,expected = arguments(m,n,k,(-79,-2,-64,3),out_strides,dtype,c_origin=origin,sensitive=sensitive)
        replay(qualified,tmp_path / f"case-{index}",values,expected)
