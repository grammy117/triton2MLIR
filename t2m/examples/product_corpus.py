"""Prepare reproducible source/ABI/reference cases without running conversion."""

import argparse
import json
from pathlib import Path

import numpy as np

from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer


def prepare(out):
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    options = {"memory_profile": "guarded-buffer", "address_policy": "source-words"}

    def add(name, source, abi, examples=(), **policy):
        work = out / name
        work.mkdir()
        (work / "source.ttir").write_text(source)
        (work / "abi.json").write_text(json.dumps(abi.as_dict(), indent=2) + "\n")
        saved = []
        for index, (arrays, reference) in enumerate(examples):
            np.savez(work / f"inputs{index}.npz", **{f"in{i}": a for i, a in enumerate(arrays)})
            np.savez(work / f"reference{index}.npz", **{f"out{i}": a for i, a in enumerate(reference)})
            saved.append({"inputs": f"{name}/inputs{index}.npz", "reference": f"{name}/reference{index}.npz"})
        cases.append({"id": name, "source": f"{name}/source.ttir", "abi": f"{name}/abi.json",
                      "options": options, "examples": saved, **policy})

    def captured(kernel, signature, constants):
        return capture_ttir(kernel, {**signature, **{key: "constexpr" for key in constants}}, constants)

    def buffers(names, shapes, dtypes):
        return tuple(TensorBuffer(name, shape, "write" if name.startswith("out") else "read", dtype)
                     for name, shape, dtype in zip(names, shapes, dtypes, strict=True))

    from t2m.examples import tensor_kernels
    shape_source = Path(tensor_kernels.__file__).with_name("shape_pipeline.ttir").read_text()
    abi = TensorABI("shape_pipeline", buffers(("x", "y", "out"), ((4,), (4,), (8,)), ("i32",) * 3))
    examples = []
    for shift in (0, 11):
        x, y = np.arange(4, dtype=np.int32) + shift, np.arange(4, dtype=np.int32) * -3
        examples.append(([x, y], [2 * np.concatenate((x, y)) + x[0] + 1]))
    add("shape-values", shape_source, abi, examples)

    from t2m.examples.tile_gather_kernels import tile_gather_values
    source = captured(tile_gather_values, {"x": "*i32", "indices": "*i64", "out": "*i32"},
                      {"AXIS": 1, "R": 4, "S": 8, "OR": 4, "OS": 4})
    abi = TensorABI("tile_gather_values", buffers(("x", "indices", "out"), ((4, 8), (4, 4), (4, 4)), ("i32", "i64", "i32")))
    examples = []
    for shift in (0, 3):
        x = np.arange(32, dtype=np.int32).reshape(4, 8) * 17
        indices = ((np.indices((4, 4)).sum(axis=0) * 3 + shift) % 8).astype(np.int64)
        examples.append(([x, indices], [np.take_along_axis(x, indices, axis=1)]))
    add("tile-gather", source, abi, examples)

    from t2m.examples.elementwise_map_kernels import (
        elementwise_map_values,
        packed_map_values,
        pointer_load_map,
    )
    source = captured(elementwise_map_values, {"x": "*i32", "y": "*i32", "out_a": "*i32", "out_b": "*i32",
                      "n": "i32", "iterations": "i32"}, {"B": 8, "WHILE_LOOP": True})
    abi = TensorABI("elementwise_map_values", buffers(("x", "y", "out_a", "out_b"), (("n",),) * 4, ("i32",) * 4),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                                                   RuntimeScalar("iterations", "i32", 5, bounds=(0, 4))))
    examples = []
    for n, count in ((13, 3), (0, 0), (31, 4)):
        x, y = np.arange(n, dtype=np.int32) - 5, np.resize(np.asarray([0, 2, -4, 0, -2, 4], np.int32), n)
        first = np.asarray([int(int(a) / int(b)) if b else -7 for a, b in zip(x, y, strict=True)], np.int32)
        for step in range(count):
            first += y * step
        examples.append(([x, y, np.asarray(n, np.int32), np.asarray(count, np.int32)], [first, x + y]))
    add("scalar-map", source, abi, examples)
    for name, kernel, names in (("packed-map", packed_map_values, ("x", "y", "out")),
                                ("memory-map", pointer_load_map, ("x", "out"))):
        source = captured(kernel, {key: "*fp32" for key in names}, {})
        abi = TensorABI(kernel.__name__, buffers(names, ((8,),) * len(names), ("f32",) * len(names)))
        examples = []
        if name == "memory-map":
            for shift in (0, 3):
                x = np.arange(8, dtype=np.float32) - shift
                examples.append(([x], [x.copy()]))
        add(name, source, abi, examples)

    from t2m.examples.histogram_kernels import histogram_tiles
    source = captured(histogram_tiles, {"x": "*i64", "out": "*i32", "n": "i32", "tiles": "i32"},
                      {"B": 16, "NB": 8, "MASKED": True, "ROW_VIEW": True, "ROW_MASK": True})
    abi = TensorABI("histogram_tiles", buffers(("x", "out"), (("n",), ("tiles", 8)), ("i64", "i32")),
        grid=(GridExtent("tiles", 1),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 31)),
                                                       RuntimeScalar("tiles", "i32", 3, bounds=(0, 3))))
    examples = []
    for n, tiles in ((31, 3), (0, 0), (17, 2)):
        x = np.resize(np.asarray([0, 1, 1, 7, -1, 8, 2, (1 << 63) - 1, -(1 << 63)], np.int64), n)
        expected = np.zeros((tiles, 8), np.int32)
        for lane, value in enumerate(x):
            if lane // 16 < tiles and lane % 3 != 1 and 0 <= value < 8:
                expected[lane // 16, value] += 1
        examples.append(([x, np.asarray(n, np.int32), np.asarray(tiles, np.int32)], [expected]))
    add("histogram", source, abi, examples)

    from t2m.examples.integer_product_kernels import high_product_values
    source = captured(high_product_values, {"x": "*i64", "y": "*i64", "out": "*i64", "n": "i32"}, {"B": 8})
    abi = TensorABI("high_product_values", buffers(("x", "y", "out"), (("n",),) * 3, ("i64",) * 3),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),))
    examples = []
    mask = (1 << 64) - 1
    for n in (13, 0, 31):
        words = np.asarray([0, 1, mask, mask - 1, 1 << 63, (1 << 63) - 1, 0x12345678], np.uint64)
        x, y = np.resize(words, n).view(np.int64), np.resize(np.roll(words, 3), n).view(np.int64)
        reference = np.asarray([((int(a) & mask) * (int(b) & mask)) >> 64
                                for a, b in zip(x, y, strict=True)], np.uint64).view(np.int64)
        examples.append(([x, y, np.asarray(n, np.int32)], [reference]))
    add("high-product", source, abi, examples)

    kernel = tensor_kernels.blockptr_matmul
    source = captured(kernel, {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
                      {"M": 19, "N": 17, "K": 23, "BM": 16, "BN": 16, "BK": 16})
    abi = TensorABI(kernel.__name__, buffers(("x", "y", "out"), ((19, 23), (23, 17), (19, 17)), ("f32",) * 3), grid=(2, 2))
    examples = []
    for shift in (0, 1):
        x = ((np.arange(19 * 23).reshape(19, 23) % 11) - 5 + shift).astype(np.float32) / 8
        y = ((np.arange(23 * 17).reshape(23, 17) % 7) - 3).astype(np.float32) / 8
        examples.append(([x, y], [x @ y]))
    add("block-pointer-dot", source, abi, examples)

    kernel = tensor_kernels.tuple_scan
    source = captured(kernel, {"a": "*i32", "b": "*i32", "out_a": "*i32", "out_b": "*i32"},
                      {"R": 2, "S": 8, "AXIS": 1, "REVERSE": False})
    abi = TensorABI(kernel.__name__, buffers(("a", "b", "out_a", "out_b"), ((2, 8),) * 4, ("i32",) * 4))
    examples = []
    for shift in (0, 1):
        a = (np.arange(16).reshape(2, 8) % 4 - 1 + shift).astype(np.int32)
        b = (np.arange(16).reshape(2, 8) % 5 - 2).astype(np.int32)
        out_a, out_b = a.copy(), b.copy()
        for column in range(1, 8):
            out_a[:, column] = a[:, column] * out_a[:, column - 1]
            out_b[:, column] = a[:, column] * out_b[:, column - 1] + b[:, column]
        examples.append(([a, b], [out_a, out_b]))
    add("tuple-scan", source, abi, examples)

    kernel = tensor_kernels.row_softmax
    source = captured(kernel, {"x": "*fp32", "out": "*fp32"}, {"N": 13, "BLOCK": 16})
    abi = TensorABI(kernel.__name__, buffers(("x", "out"), ((13,),) * 2, ("f32",) * 2))
    examples = []
    for shift in (0, 1):
        x = ((np.arange(13, dtype=np.float32) * (shift + 1)) % 11 - 5) / 8
        values = np.exp(x - np.max(x))
        examples.append(([x], [values / np.sum(values)]))
    add("softmax-reduction", source, abi, examples, rtol=1e-5, atol=2e-6)

    source = '''module {tt.func public @roundtrip(%x:!tt.ptr<i32>,%out:!tt.ptr<i32>) {
      %bits=tt.ptr_to_int %x:!tt.ptr<i32> -> i64
      %ptr=tt.int_to_ptr %bits:i64 -> !tt.ptr<i32>
      %v=tt.load %ptr:!tt.ptr<i32>
      tt.store %out,%v:!tt.ptr<i32>
      tt.return
    }}'''
    abi = TensorABI("roundtrip", buffers(("x", "out"), ((1,),) * 2, ("i32",) * 2))
    add("pointer-roundtrip", source, abi, [([np.asarray([42], np.int32)], [np.asarray([42], np.int32)])])
    source = '''module {tt.func public @assembly(%x:!tt.ptr<i32>,%out:!tt.ptr<i32>) {
      %v=tt.load %x:!tt.ptr<i32>
      %w="tt.elementwise_inline_asm"(%v) <{asm_string="add.u32 $0, $1, 1;",
        constraints="=r,r", pure=true, packed_element=1:i32}>:(i32)->i32
      tt.store %out,%w:!tt.ptr<i32>
      tt.return
    }}'''
    abi = TensorABI("assembly", buffers(("x", "out"), ((1,),) * 2, ("i32",) * 2))
    add("source-assembly", source, abi, [([np.asarray([42], np.int32)], [np.asarray([43], np.int32)])])
    path = out / "manifest.json"
    path.write_text(json.dumps({"schema": "t2m.product-corpus.v1", "cases": cases}, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(prepare(parser.parse_args().out))
