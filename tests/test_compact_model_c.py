"""Real three-kernel PyTorch compaction through checked C result transport."""

import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_compact_model import REQUIRED, parent  # noqa: F401
from test_compact_model import selected as captured_selection

from t2m.bundle import verify_bundle

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing source/model/Merlin tools required")
selected = captured_selection

DRIVER = r'''
#include "owned_result_views.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern int merlin_owned_result_invoke_checked(void **,MerlinOwnedResultView *,size_t);
typedef struct {void *allocated,*aligned;intptr_t offset;} D0;
typedef struct {void *allocated,*aligned;intptr_t offset,sizes[1],strides[1];} D1;
int main(int argc,char **argv) {
  struct rlimit no_core={0,0}; setrlimit(RLIMIT_CORE,&no_core);
  setbuf(stdout,NULL);
  if(argc<2 || argc>34) return 1;
  int32_t weight=(int32_t)strtoll(argv[1],NULL,10), x[32], before[32];
  for(int i=2;i<argc;i++) x[i-2]=(int32_t)strtoll(argv[i],NULL,10);
  memcpy(before,x,(argc-2)*sizeof(int32_t));
  D0 parameter={&weight,&weight,0}; D1 input={x,x,0,{argc-2},{1}};
  void *arguments[]={&parameter,&input};
  for(int repeat=0;repeat<2;repeat++) {
    if(!merlin_owned_results_begin(T2M_MAX_BYTES,T2M_MAX_ALLOCATIONS)) return 2;
    MerlinOwnedResultView views[7];
    int status=merlin_owned_result_invoke_checked(arguments,views,T2M_MAX_OUTPUT_BYTES);
    if(status) {fputs(merlin_owned_result_error(status),stderr);return 3;}
    void *outputs[7];
    for(int i=0;i<7;i++) {
      outputs[i]=malloc(views[i].bytes ? views[i].bytes : 1);
      if(!outputs[i] || merlin_owned_result_copy(&views[i],outputs[i],views[i].bytes)) return 4;
    }
    merlin_owned_results_end();
    for(int i=0;i<7;i++) {
      uint64_t rank=views[i].rank, width=views[i].element_bytes, bytes=views[i].bytes;
      if(fwrite(&rank,8,1,stdout)!=1) return 5;
      for(size_t axis=0;axis<rank;axis++) {
        uint64_t size=views[i].sizes[axis]; if(fwrite(&size,8,1,stdout)!=1) return 6;
      }
      if(fwrite(&width,8,1,stdout)!=1 || fwrite(&bytes,8,1,stdout)!=1 ||
         fwrite(outputs[i],1,views[i].bytes,stdout)!=views[i].bytes) return 7;
      free(outputs[i]);
    }
    if(memcmp(before,x,(argc-2)*sizeof(int32_t)) || weight!=(int32_t)strtoll(argv[1],NULL,10)) return 8;
  }
  return 0;
}
'''


@pytest.fixture(scope="module")
def checked_c(selected, tmp_path_factory):
    _root, _parent, _pack, package = selected
    verify_bundle(package)
    interface = json.loads((package / "consumer/interface.json").read_text())
    for name in ("src/merlin/llvmlower/owned_result_c.py", "merlin/runtime/abi/owned_result_views.c",
                 "merlin/runtime/abi/owned_result_views.h"):
        assert name in interface["consumer_source_sha256"]
    signature = json.loads((package / "consumer/compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v6" and len(signature["outputs"]) == 7
    policy = signature["allocation_policy"]
    root = tmp_path_factory.mktemp("compact-checked-c")
    source, executable = root / "driver.c", root / "driver"
    source.write_text(DRIVER)
    library = package / "consumer/host/model.so"
    runtime = Path(os.environ["T2M_MERLIN_ROOT"]) / "merlin/runtime/abi"
    done = subprocess.run(
        [os.environ["T2M_MERLIN_CLANG"], "-std=c11", "-Wall", "-Wextra", "-O2",
         f'-DT2M_MAX_BYTES={policy["max_bytes"]}',
         f'-DT2M_MAX_ALLOCATIONS={policy["max_allocations"]}',
         f'-DT2M_MAX_OUTPUT_BYTES={policy["max_output_bytes"]}',
         "-I", str(runtime), str(source), str(library),
         "-Wl,-rpath," + str(library.parent), "-o", str(executable)],
        capture_output=True, text=True, timeout=30, check=False,
        env=dict(os.environ, TMPDIR=str(root)),
    )
    assert done.returncode == 0, done.stderr
    return executable, library, hashlib.sha256(library.read_bytes()).hexdigest(), package


CASES = [([0, -1], 0), ([4, -7, 2], 0), ([1, -1] * 15 + [7], 0),
         (list(range(31)), 100),
         ([0, -1, 1, -(1 << 31), (1 << 31) - 1] * 6 + [7], 1),
         ([-9, 3, 8, -2, 0, 5, -1, 6, 1, -3, 7, 2, -8], -3)]


@pytest.mark.parametrize("words,weight", CASES)
def test_c_transports_every_real_ttir_model_result_without_shape_prediction(checked_c, tmp_path, words, weight):
    executable, library, digest, package = checked_c
    done = subprocess.run([str(executable), str(weight), *map(str, words)],
                          capture_output=True, timeout=10, check=False)
    assert done.returncode == 0, (done.returncode, done.stderr)
    (tmp_path / "results.bin").write_bytes(done.stdout)
    x, parameter = np.asarray(words, np.int32), np.asarray(weight, np.int32)
    original = np.add(x, parameter, dtype=np.int32)
    first = original[original > 0].copy()
    expected = (first, first, original, x, np.asarray([len(first)], np.int64), parameter, first[::2])
    cursor = 0

    def word():
        nonlocal cursor
        value = int(np.frombuffer(done.stdout[cursor:cursor + 8], dtype=np.uint64)[0])
        cursor += 8
        return value

    for _repeat in range(2):
        for reference in expected:
            rank = word()
            shape = tuple(word() for _ in range(rank))
            width, count = word(), word()
            assert shape == reference.shape and width == reference.itemsize and count == reference.nbytes
            assert done.stdout[cursor:cursor + count] == reference.tobytes()
            cursor += count
    assert cursor == len(done.stdout)
    assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
    verify_bundle(package)
    np.savez(tmp_path / "references.npz", **{f"out{i}": value for i, value in enumerate(expected)})


@pytest.mark.parametrize("capacity", [1, 32])
def test_c_invocation_retains_actual_model_domain_assertions(checked_c, tmp_path, capacity):
    executable, _library, _digest, _package = checked_c
    done = subprocess.run([str(executable), "0", *map(str, range(capacity))],
                          capture_output=True, timeout=10, check=False)
    (tmp_path / "abort.json").write_text(json.dumps({
        "returncode": done.returncode, "stdout": done.stdout.decode(errors="replace"),
        "stderr": done.stderr.decode(errors="replace"),
    }, sort_keys=True) + "\n")
    assert done.returncode == -signal.SIGABRT
    assert b"exported shape domain:" in done.stdout + done.stderr
