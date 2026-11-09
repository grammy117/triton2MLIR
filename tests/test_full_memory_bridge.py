"""Closed integration probe of the full shared pipeline, not generic admission."""

import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

from t2m.frontend import capture_ttir
from t2m.upstream import lower_with_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN, PYTHON, CLANG = (os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not all((OPT, NATIVE, MERLIN, PYTHON, CLANG)), reason="existing full-stage/Merlin toolchains required")


BUILD = '''
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text
from xdsl.dialects import arith,bufferization,func,memref
from xdsl.dialects.builtin import ArrayAttr,DenseArrayBase,DictionaryAttr,IndexType,IntegerAttr,MemRefType,ModuleOp,TensorType,UnitAttr,f32,i1,i32,i64
from xdsl.ir import Block,Region
from t2m.standard_ops import ForallOp,InParallelOp
from t2m.scalar_abi import RuntimeScalar,guard_scalar
root=Path(sys.argv[2])
original=next(iter(parse_mlir_text((root/'upstream/lowered.mlir').read_text()).ops))
t,b=TensorType(f32,[13]),MemRefType(f32,[13])
entry=Block(arg_types=[t,t,i32])
guard_scalar(RuntimeScalar('n','i32',1,bounds=(0,13)),entry.args[2],entry,original.location)
buffers=[]
for value in entry.args[:2]:
 op=bufferization.ToBufferOp.build(operands=[value],result_types=[b],properties={'read_only':UnitAttr()})
 entry.add_op(op);buffers.append(op.memref)
owned=memref.AllocOp.build(operands=[[],[]],result_types=[b])
entry.add_ops([owned,memref.CopyOp(buffers[1],owned.memref)])
body=Block(arg_types=[IndexType()])
constants=[]
for n in (0,1,4):
 op=arith.ConstantOp.from_int_and_width(n,i32);body.add_op(op);constants.append(op.result)
zero,one,four=constants
pid=arith.IndexCastOp(body.args[0],i32);body.add_op(pid)
# Source ABI: original arguments, then x/y/z counts, then x/y/z IDs.
values=[buffers[0],entry.args[2],owned.memref,four,one,one,pid.result,zero,zero]
assert len(values)==len(original.body.block.args)
mapper=dict(zip(original.body.block.args,values,strict=True))
for source in original.body.block.ops:
 if not isinstance(source,func.ReturnOp): body.add_op(source.clone(value_mapper=mapper))
body.add_op(InParallelOp.build(regions=[Region(Block())]))
entry.add_op(ForallOp.build(operands=[[],[],[],[]],result_types=[[]],regions=[Region(body)],properties={
 'staticLowerBound':DenseArrayBase.from_list(i64,[0]),'staticUpperBound':DenseArrayBase.from_list(i64,[4]),'staticStep':DenseArrayBase.from_list(i64,[1])}))
returned=bufferization.ToTensorOp(owned.memref,restrict=True,writable=True)
entry.add_ops([returned,func.ReturnOp(returned.tensor)])
module=ModuleOp([func.FuncOp('forward',([t,t,i32],[t]),Region(entry))]);module.verify()
assert all(op.name!='builtin.unregistered' for op in module.walk())
function=next(iter(module.ops))
function.properties['arg_attrs']=ArrayAttr(DictionaryAttr({'bufferization.writable':IntegerAttr(0,i1)} if isinstance(t,TensorType) else {}) for t in function.function_type.inputs)
from merlin.llvmlower.passes_xdsl import add_c_interface
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.codegen import build_host_shared
from t2m.consumer_worker import COMPILER_FEATURES, tensor_pipeline
add_c_interface(module)
prepared=text(module,generic=True);(root/'prepared.mlir').write_text(prepared)
llvm=lower_to_llvm_ir(prepared,workdir=root/'lowering',pipeline=tensor_pipeline(),timeout=30,features=COMPILER_FEATURES,retain_llvm_dialect=True)
(root/'model.ll').write_text(llvm);build_host_shared(root/'model.ll',root/'model.so')
'''

EXECUTE = '''
from pathlib import Path
import hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from merlin.llvmlower.abi import HostModel,ScalarArg
root=Path(sys.argv[2]);library=root/'model.so'
model=HostModel.load(str(library.resolve()))
x=np.arange(13,dtype=np.float32);old=np.arange(13,dtype=np.float32)*7+99
before=[x.tobytes(),old.tobytes()]
for n in (0,1,7,13):
 y=np.zeros(13,dtype=np.float32)
 model([(x.ctypes.data,x.shape),(old.ctypes.data,old.shape),ScalarArg(n,'i32'),(y.ctypes.data,y.shape)])
 expected=old.copy();expected[:n]=x[:n]+2
 np.testing.assert_array_equal(y,expected)
 assert [x.tobytes(),old.tobytes()]==before
print(json.dumps({'scope':'closed-full-stage-Merlin-probe','values':[0,1,7,13],'inputs_preserved':True,
 'portable_conversion_complete':False,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}))
'''


def worker(code, root):
    import t2m
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root),
        PYTHONPATH=str(Path(t2m.__file__).parent.parent), MERLIN_COMPILER_PYTHON=NATIVE, MERLIN_CLANG=CLANG,
        MERLIN_COMPILE_TIMEOUT_S="30")
    process = subprocess.Popen([PYTHON, "-c", code, MERLIN, str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("bounded full-stage compiler/evaluator probe exceeded 60 seconds")
    assert process.returncode == 0, stderr[-5000:]
    return stdout


def test_full_shared_memory_lowering_can_use_merlin_without_rebuilding_payload(tmp_path):
    from t2m.examples.tensor_kernels import runtime_prefix
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_prefix, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    stage = lower_with_shared(source, tmp_path / "upstream", optimizer=OPT, stage="full")
    assert stage["stage"] == "full"
    worker(BUILD, tmp_path)  # No evaluator data/reference supplied to compilation.
    report = json.loads(worker(EXECUTE, tmp_path))
    assert report["values"] == [0, 1, 7, 13]
    assert report["inputs_preserved"] is True
    assert report["portable_conversion_complete"] is False
