#!/usr/bin/env python3
"""Fetch one content-pinned, prebuilt standalone optimizer; no backend install.

The upstream Meta/Microsoft repositories publish no optimizer release assets.
AMD's wheel packages the shared optimizer as a standalone executable. This is
an explicitly identified distribution, not a claim that it matches Meta HEAD.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ASSET = "triton_xdna-3.8.0.2026110704+7180fcc-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "3627dc579a7d25219fb1afba736a491042248397a088a0d1e38e808529845fa4"
BINARY_SHA256 = "2af44b4536b2dffbdb90b877b04e97cd2b10fe333020606fa4a1814b3c34926d"
MEMBER = "triton/triton_shared/triton-shared-opt"
URL = "https://github.com/amd/Triton-XDNA/releases/download/latest-wheels/" + ASSET.replace("+", "%2B")
SOURCE = {
    "distribution": "https://github.com/amd/Triton-XDNA",
    "revision": "7180fccfa0283456eb083a5718104da8156ae50b",
    "triton_submodule": "c8e7cc932320d05d08f39ffc99bfda32dc4f22fa",
    "triton_shared_submodule": "0c27923cbcb51f1565ae39c2de0e2245babb6067",
    "build_source_attestation": "not-independently-reproduced",
}


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    binary, wheel = destination / "triton-shared-opt", destination / ASSET
    if binary.exists():
        raise FileExistsError(binary)
    digest, size, start = hashlib.sha256(), 0, time.monotonic()
    # 408 MB download, 478 MB executable; no compilation or package installation.
    with urllib.request.urlopen(URL, timeout=30) as response, wheel.open("xb") as stream:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > 500_000_000 or time.monotonic() - start > 180:
                raise ValueError("optimizer download exceeded its size/time budget")
            digest.update(chunk)
            stream.write(chunk)
    if digest.hexdigest() != WHEEL_SHA256:
        raise ValueError("download does not match the pinned wheel digest")
    with zipfile.ZipFile(wheel) as archive:
        contents = archive.read(MEMBER)
    if hashlib.sha256(contents).hexdigest() != BINARY_SHA256:
        raise ValueError("optimizer does not match the pinned executable digest")
    with binary.open("xb") as stream:
        stream.write(contents)
    binary.chmod(0o755)
    report = {"schema": "t2m.prebuilt-tool.v1", "url": URL, "source": SOURCE,
              "wheel_sha256": WHEEL_SHA256, "binary_sha256": BINARY_SHA256,
              "binary_bytes": len(contents), "wheel_bytes": size,
              "backend_installed": False, "native_build_performed": False}
    (destination / "toolchain.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    wheel.unlink()  # Remove only the temporary wheel this invocation created.
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".scratch/shared-opt"))
    print(json.dumps(fetch(parser.parse_args().out), indent=2))
