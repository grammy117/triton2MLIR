"""C stdout transport for fail-stop diagnostics in owned native workers.

Upstream cf-to-LLVM emits puts followed by abort. Disabling buffering in the
isolated evaluator process preserves that output before abort terminates it.
This changes no portable IR, compiler input or failure/termination semantics.
"""

import ctypes


def prepare_stdout():
    import locale
    locale.setlocale(locale.LC_NUMERIC,"C")
    libc = ctypes.CDLL(None)
    try:
        stdout = ctypes.c_void_p.in_dll(libc, "stdout")
        setbuf = libc.setbuf
    except (ValueError, AttributeError) as error:
        raise RuntimeError("native diagnostic transport requires libc stdout/setbuf") from error
    setbuf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    setbuf.restype = None
    setbuf(stdout, None)
