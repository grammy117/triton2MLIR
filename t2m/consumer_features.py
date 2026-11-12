"""Select existing consumer capabilities from executable types and operations.

Selection grants no legality, numeric, alias or target permission. Merlin's
selected implementations independently validate every supported body.
This module also loads directly in compiler-only workers without importing
the frontend package or a source-language implementation.
"""

COMPILER_FEATURES = frozenset({"reuse_loop_tensor_destinations"})


def select_features(module):
    if __package__:
        from .bfloat_math import requires_emulation
        from .fp8_math import requires_widening, requires_narrowing
        from .native import requires_integer_atomic_regions
    else:
        from bfloat_math import requires_emulation
        from fp8_math import requires_widening, requires_narrowing
        from native import requires_integer_atomic_regions

    features = set(COMPILER_FEATURES)
    if any("bf16" in str(value.type) for op in module.walk() for value in (*op.operands, *op.results)):
        features.add("preserve_bfloat_transport")
    if requires_emulation(module.walk()):
        features.add("emulate_bfloat_elementwise")
    if requires_widening(module.walk()):
        features.add("lower_fp8_widening")
    if requires_narrowing(module.walk()):
        features.add("lower_fp8_narrowing")
    if any(op.name == "vector.print" for op in module.walk()):
        features.add("lower_scalar_diagnostics")
    elif any(op.name == "cf.assert" and "\x00" in op.msg.data for op in module.walk()):
        features.add("lower_length_aware_diagnostics")
    regions = [op for op in module.walk() if op.name == "memref.generic_atomic_rmw"]
    integer_regions = requires_integer_atomic_regions(module)
    if integer_regions:
        from merlin.llvmlower.atomic_integer_regions import ensure_registered
        ensure_registered()
        features.add("lower_integer_atomic_regions")
    if any(str(op.results[0].type) in ("f16", "bf16", "f32", "f64") for op in regions):
        features.add("lower_atomic_word_views")
    elif regions and not integer_regions:
        features.add("lower_integer_compare_exchange")
    return frozenset(features)
