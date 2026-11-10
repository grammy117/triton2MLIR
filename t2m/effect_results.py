"""Typed, bounded numeric references accompanying observable kernel outcomes.

References are evaluator data. Exact word policies preserve NaN payloads,
signed zero and integer low bits; value policies reuse the existing logical
tensor comparison, including BF16's explicit uint16 transport.
"""

import math
from pathlib import Path
import zipfile

import numpy as np

from .runtime_shapes import resolve_shapes
from .tensor_storage import DTYPES,assert_tensor_close

ARCHIVE = "reference.npz"
LIMIT = 64*1024*1024


def _tolerance(value):
    if type(value) not in (int,float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_policy(results):
    if (not isinstance(results,dict) or set(results) != {"archive","comparisons"}
            or results["archive"] != ARCHIVE or not isinstance(results["comparisons"],list)
            or not results["comparisons"]):
        raise ValueError("mixed result references require reference.npz and complete per-result comparisons")
    for policy in results["comparisons"]:
        if not isinstance(policy,dict):
            raise ValueError("result comparison must be an explicit word or value policy")
        if policy == {"kind":"words"}:
            continue
        if (set(policy) != {"kind","rtol","atol"} or policy["kind"] != "values"
                or any(not _tolerance(policy[key]) for key in ("rtol","atol"))):
            raise ValueError("value comparison requires finite nonnegative explicit tolerances")
    return results


def reference_members(reference):
    return ("reference.json",ARCHIVE) if reference["schema"] == "t2m.effect-reference.v2" else ("reference.json",)


def load_archive(path,count):
    """Validate uncompressed sizes and NPY headers before NumPy allocates."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT+1024*1024:
        raise ValueError("result reference archive must be a bounded regular file")
    names = {f"out{i}.npy" for i in range(count)}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != count or {item.filename for item in members} != names:
            raise ValueError("result archive must contain exactly the complete out0, out1, ... roster")
        total = 0
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1,0),(2,0)):
                    raise ValueError("result archive uses an unsupported NPY header version")
                reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape,_,dtype = reader(stream,max_header_size=16384)
                if (dtype.hasobject or dtype.fields is not None or dtype not in tuple(np.dtype(v) for v in DTYPES.values())
                        or any(type(n) is not int or n < 0 for n in shape)):
                    raise ValueError("result archive requires supported primitive tensor carriers")
                size = math.prod(shape)*dtype.itemsize
                total += size
                if total > LIMIT or member.file_size != stream.tell()+size:
                    raise ValueError("result archive data/header exceeds its complete bounded tensor storage")
    with np.load(path,allow_pickle=False) as arrays:
        return tuple(arrays[f"out{i}"].copy() for i in range(count))


def validate_results(spec,inputs,arrays,comparisons):
    if len(arrays) != len(spec.outputs) or len(comparisons) != len(spec.outputs) or not spec.outputs:
        raise ValueError("mixed references must bind every actual declared tensor result")
    if sum(np.asarray(value).nbytes for value in arrays) > LIMIT:
        raise ValueError("result references exceed the 64MiB tensor budget")
    shapes = resolve_shapes(spec.runtime_shapes(),inputs) if spec.dynamic else tuple(value.shape for value in spec.outputs)
    for index,(array,value,shape,policy) in enumerate(zip(arrays,spec.outputs,shapes,comparisons,strict=True)):
        array = np.asarray(array)
        if array.dtype != DTYPES[value.dtype] or tuple(array.shape) != tuple(shape):
            raise ValueError(f"result out{index} differs from the actual typed runtime shape/ABI")
        if value.dtype == "i1" and np.any(array.view(np.uint8) > 1):
            raise ValueError("result reference has noncanonical Boolean storage")
        if value.dtype.startswith("i") and policy["kind"] == "values" and (policy["rtol"] or policy["atol"]):
            raise ValueError("integer result comparisons require exact values or words")
    return tuple(np.asarray(value) for value in arrays)


def compare_results(actual_path,reference_path,reference,spec,inputs):
    policies = validate_policy(reference["results"])["comparisons"]
    count = len(spec.outputs)
    expected = validate_results(spec,inputs,load_archive(reference_path,count),policies)
    actual = validate_results(spec,inputs,load_archive(actual_path,count),policies)
    for index,(left,right,value,policy) in enumerate(zip(actual,expected,spec.outputs,policies,strict=True)):
        try:
            if policy["kind"] == "words":
                if left.tobytes(order="C") != right.tobytes(order="C"):
                    raise ValueError(f"actual result out{index} differs from its exact word reference")
            else:
                assert_tensor_close(left,right,value.dtype,rtol=policy["rtol"],atol=policy["atol"])
        except AssertionError as error:
            raise ValueError(f"actual result out{index} differs from its numeric value reference") from error
    return count
