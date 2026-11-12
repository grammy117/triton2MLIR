"""Original layout-free TTIR fixture with scalar/dense IEEE sentinel literals.

The supplied TTIR is the source program, rather than output from a Python
float round trip. Its loop reuses loaded tiles and preserves untouched words.
"""


def literal_words(dtype):
    bits, nan = {
        "f16": (16, 0x7E00),
        "bf16": (16, 0x7FC0),
        "f32": (32, 0x7FC00000),
        "f64": (64, 0x7FF8000000000000),
    }[dtype]
    sign = 1 << (bits - 1)
    return (
        bits,
        (nan - 1) | sign,
        [nan | 1, sign, nan - 1, 1, nan | sign | 3, 0, nan, nan | 7],
    )


def source(dtype):
    _, sentinel, words = literal_words(dtype)
    tile = f"tensor<8x{dtype}>"
    pointers = f"tensor<8x!tt.ptr<{dtype}>>"
    values = ",".join(hex(word) for word in words)
    return f"""module {{
      tt.func public @literal_state(%x:!tt.ptr<{dtype}>,%state:!tt.ptr<{dtype}>,
          %snapshot:!tt.ptr<{dtype}>,%n:i32,%steps:i32) {{
        %pid=tt.get_program_id x:i32
        %b=arith.constant 8:i32
        %zero=arith.constant 0:i32
        %one=arith.constant 1:i32
        %start=arith.muli %pid,%b:i32
        %starts=tt.splat %start:i32->{"tensor<8xi32>"}
        %range=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
        %offsets=arith.addi %starts,%range:tensor<8xi32>
        %size=tt.splat %n:i32->tensor<8xi32>
        %active=arith.cmpi slt,%offsets,%size:tensor<8xi32>
        %xp=tt.splat %x:!tt.ptr<{dtype}>->{pointers}
        %sp=tt.splat %state:!tt.ptr<{dtype}>->{pointers}
        %op=tt.splat %snapshot:!tt.ptr<{dtype}>->{pointers}
        %xa=tt.addptr %xp,%offsets:{pointers},tensor<8xi32>
        %sa=tt.addptr %sp,%offsets:{pointers},tensor<8xi32>
        %oa=tt.addptr %op,%offsets:{pointers},tensor<8xi32>
        %literal=arith.constant {hex(sentinel)}:{dtype}
        %fill=tt.splat %literal:{dtype}->{tile}
        %table=arith.constant dense<[{values}]>:{tile}
        %loaded=tt.load %xa,%active,%fill:{pointers}
        %original=tt.load %sa,%active,%fill:{pointers}
        %ones=arith.constant dense<1>:tensor<8xi32>
        %zeroes=arith.constant dense<0>:tensor<8xi32>
        %parity=arith.andi %offsets,%ones:tensor<8xi32>
        %even=arith.cmpi eq,%parity,%zeroes:tensor<8xi32>
        %result=scf.for %i=%zero to %steps step %one iter_args(%carry=%original)->({tile}) : i32 {{
          %position=arith.addi %i,%pid:i32
          %odd=arith.andi %position,%one:i32
          %choose=arith.cmpi eq,%odd,%zero:i32
          %next=scf.if %choose -> ({tile}) {{
            %updated=arith.select %even,%table,%carry:tensor<8xi1>,{tile}
            scf.yield %updated:{tile}
          }} else {{
            %updated=arith.select %even,%carry,%loaded:tensor<8xi1>,{tile}
            scf.yield %updated:{tile}
          }}
          scf.yield %next:{tile}
        }}
        %saved=arith.select %even,%fill,%original:tensor<8xi1>,{tile}
        tt.store %sa,%result,%active:{pointers}
        tt.store %oa,%saved,%active:{pointers}
        tt.return
      }}
    }}"""
