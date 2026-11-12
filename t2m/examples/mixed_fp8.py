"""Typed TTIR fixture for independent per-cast software numeric contracts."""

from t2m.fp8_casts import POLICIES, SOURCE_POLICY


def source(target="f8E5M2", mode="rtne"):
    rounds, shifted, stores = [], [], []
    for i, policy in enumerate(POLICIES):
        rounds.append(f'''%q{i}="tt.fp_to_fp"(%x{i}) {{rounding={int(mode == "rtne")}:i32,
          {SOURCE_POLICY}="{policy}",prov.family="quantize",prov.op="cast",prov.region_id="policy{i}"}}
          :(tensor<32xf32>)->tensor<32x{target}>
        %r{i}=arith.extf %q{i}:tensor<32x{target}> to tensor<32xf32>''')
        shifted.append(f'%v{i}=arith.addf %a{i},%dv:tensor<32xf32>')
        stores.append(f'''%p{i}=tt.splat %out{i}:!tt.ptr<f32>->tensor<32x!tt.ptr<f32>>
          %oa{i}=tt.addptr %p{i},%lane:tensor<32x!tt.ptr<f32>>,tensor<32xi32>
          tt.store %oa{i},%result#{i},%mask:tensor<32x!tt.ptr<f32>>''')
    types = ','.join(['tensor<32xf32>'] * 3)
    return f'''module {{
      tt.func private @quantize(%x0:tensor<32xf32>,%x1:tensor<32xf32>,%x2:tensor<32xf32>)->({types}) {{
        {chr(10).join(rounds)}
        tt.return %r0,%r1,%r2:{types}
      }}
      tt.func @mixed_fp8(%x:!tt.ptr<f32>,%out0:!tt.ptr<f32>,%out1:!tt.ptr<f32>,%out2:!tt.ptr<f32>,
                        %raw:!tt.ptr<f32>,%n:i32,%steps:i32,%delta:f32) {{
        %pid=tt.get_program_id x:i32
        %block=arith.constant 32:i32
        %base=arith.muli %pid,%block:i32
        %range=tt.make_range {{start=0:i32,end=32:i32}}:tensor<32xi32>
        %b=tt.splat %base:i32->tensor<32xi32>
        %lane=arith.addi %b,%range:tensor<32xi32>
        %nv=tt.splat %n:i32->tensor<32xi32>
        %mask=arith.cmpi slt,%lane,%nv:tensor<32xi32>
        %xp=tt.splat %x:!tt.ptr<f32>->tensor<32x!tt.ptr<f32>>
        %xa=tt.addptr %xp,%lane:tensor<32x!tt.ptr<f32>>,tensor<32xi32>
        %zero=arith.constant dense<0.0>:tensor<32xf32>
        %value=tt.load %xa,%mask,%zero:tensor<32x!tt.ptr<f32>>
        %dv=tt.splat %delta:f32->tensor<32xf32>
        %start=arith.constant 0:index
        %one=arith.constant 1:index
        %limit=arith.index_cast %steps:i32 to index
        %result:3=scf.for %i=%start to %limit step %one
          iter_args(%a0=%value,%a1=%value,%a2=%value)->({types}) {{
          {chr(10).join(shifted)}
          %next:3=tt.call @quantize(%v0,%v1,%v2):({types})->({types})
          scf.yield %next#0,%next#1,%next#2:{types}
        }}
        {chr(10).join(stores)}
        %rp=tt.splat %raw:!tt.ptr<f32>->tensor<32x!tt.ptr<f32>>
        %ra=tt.addptr %rp,%lane:tensor<32x!tt.ptr<f32>>,tensor<32xi32>
        tt.store %ra,%value,%mask:tensor<32x!tt.ptr<f32>>
        tt.return
      }}
    }}'''


def reference(values, target, mode, steps, delta):
    import numpy as np
    from t2m.examples.fp8_cast_reference import reference as rounded
    from t2m.tensor_storage import comparison_values
    results = [values.copy() for _ in POLICIES]
    with np.errstate(invalid="ignore", over="ignore"):
        for _ in range(steps):
            results = [comparison_values(rounded(value + np.float32(delta), "f32", target, mode, policy), target).astype(np.float32)
                       for value, policy in zip(results, POLICIES, strict=True)]
    return [*results, values.copy()]
