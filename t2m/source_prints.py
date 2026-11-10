"""Logical TTIR print events expanded to standard scalar/string diagnostics.

Triton f797708c PrintOpToLLVM.cpp and test_subprocess.py define per-operand
coordinates, signedness, hex words and prefix formatting. Physical duplicate
threads/order are abstracted to one row-major logical event per element.
"""

from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_vector import literal, scalar


def contract(op, source_id):
    def refuse(message):
        raise ConversionError(message,source_id=source_id,opcode="tt.print")
    attrs = {**op.attributes,**op.properties}
    prefix,hex_,signed = (attrs.get(key) for key in ("prefix","hex","isSigned"))
    if (not isinstance(prefix,builtin.StringAttr) or not isinstance(hex_,builtin.IntegerAttr)
            or hex_.type != builtin.i1 or not isinstance(signed,builtin.DenseArrayBase)
            or signed.elt_type != builtin.i32 or len(signed) != len(op.operands) or op.results or op.regions):
        refuse("print requires a literal prefix, Boolean hex flag and complete i32 signedness roster")
    shapes,types = [],[]
    for operand in op.operands:
        type_ = operand.type
        shape = tuple(type_.get_shape()) if isinstance(type_,builtin.TensorType) else ()
        element = type_.element_type if isinstance(type_,builtin.TensorType) else type_
        if any(n < 0 for n in shape) or element not in (builtin.i1,builtin.i8,builtin.i16,builtin.i32,builtin.i64,
                                                        builtin.f16,builtin.bf16,builtin.f32,builtin.f64):
            refuse("print requires static numeric scalar/tensor values; physical pointers need a logical-address contract")
        shapes.append(list(shape));types.append(str(element))
    return {"source_ids":[source_id],"prefix":prefix.data,"hex":bool(hex_.value.data),
        "is_signed":[value > 0 for value in signed.iter_values()],"operand_shapes":shapes,"operand_types":types,
        "semantics":"one-event-per-logical-program-operand-element",
        "formatting_profile":"logical-element-values.v1; floating-hex-is-original-typed-word; GPU-varargs-equivalence-unqualified",
        "expansion":"standard-vector.print-and-structured-scalar-formatting",
        "replication":"physical-thread-duplicates-abstracted","ordering":"operand-then-row-major-within-logical-program"}


def expand_prints(module, source_ids, grid_rank, *, logical_pids=None):
    from .shared_tensor import opcode
    reports = []
    for op in tuple(module.walk()):
        if opcode(op) != "tt.print":
            continue
        ids = source_ids(op.location)
        report = contract(op,next(iter(ids),"ttir.print"))
        report["source_ids"] = ids
        block = Block()

        def add(where, operation):
            operation.location = op.location
            operation.attributes.update({"prov.family":builtin.StringAttr("triton"),
                "prov.op":builtin.StringAttr("print"),
                "prov.source_ids":builtin.ArrayAttr(builtin.StringAttr(value) for value in ids)})
            where.add_op(operation)
            return operation.results[0] if operation.results else None

        def number(where, value, type_=builtin.i64):
            if type_ == builtin.i64 and value >= 1 << 63:
                value -= 1 << 64
            return add(where,arith.ConstantOp.from_int_and_width(value,type_))

        def string(where, value):
            add(where,literal(value))

        def show(where, value):
            add(where,scalar(value))

        def branch(where, condition, emit):
            yes = Block()
            emit(yes)
            add(yes,scf.YieldOp())
            add(where,scf.IfOp(condition,[],Region(yes),Region()))

        pids = list(logical_pids) if logical_pids is not None else []
        if logical_pids is None:
            for axis in range(3):
                if axis < grid_rank:
                    cls = builtin.UnregisteredOp.with_name("tt.get_program_id")
                    pids.append(add(block,cls.create(result_types=[builtin.i32],
                        properties={"axis":builtin.IntegerAttr(axis,builtin.i32)})))
                else:
                    pids.append(number(block,0,builtin.i32))
            parent = op.parent_op()
            while parent is not None and opcode(parent) not in ("tt.func","func.func"):
                parent = parent.parent_op()
            if parent is None or len(parent.regions) != 1 or len(parent.regions[0].blocks) != 1:
                raise ConversionError("print requires one enclosing logical kernel entry",opcode="tt.print")
            entry = parent.regions[0].block
            queries = list(block.ops)
            for query in queries:
                block.detach_op(query)
            entry.insert_ops_before(queries,entry.first_op)

        def integer64(where, value, signed=False):
            if value.type == builtin.IndexType():
                return add(where,arith.IndexCastOp(value,builtin.i64))
            if value.type == builtin.i64:
                return value
            return add(where,(arith.ExtSIOp if signed else arith.ExtUIOp)(value,builtin.i64))

        def header(where, coordinates, shape, operand=None):
            string(where,"pid (")
            for index,pid in enumerate(pids):
                if index:
                    string(where,", ")
                show(where,integer64(where,pid))
            string(where,")")
            if coordinates is not None:
                string(where," idx (")
                for index,(coordinate,size) in enumerate(zip(coordinates,shape,strict=True)):
                    if index:
                        string(where,", ")
                    value = integer64(where,coordinate)
                    for digits in range(len(str(size-1))-1,0,-1):
                        small = add(where,arith.CmpiOp(value,number(where,10**digits),"ult"))
                        branch(where,small,lambda nested:string(nested," "))
                    show(where,value)
                string(where,")")
            string(where,report["prefix"])
            if operand is not None and len(op.operands) > 1:
                string(where,f"(operand {operand}) ")

        def hex_word(where, value, width):
            integer = value
            if isinstance(value.type,builtin.AnyFloat):
                integer = add(where,arith.BitcastOp(value,builtin.IntegerType(width)))
            integer = integer64(where,integer)
            string(where,"0x")
            count = max(1,(width+3)//4)
            body = Block(arg_types=[builtin.IndexType()])
            index = integer64(body,body.args[0])
            shift = add(body,arith.SubiOp(number(body,count-1),index))
            shift = add(body,arith.MuliOp(shift,number(body,4)))
            digit = add(body,arith.ShRUIOp(integer,shift))
            digit = add(body,arith.AndIOp(digit,number(body,15)))
            for nibble in range(16):
                match = add(body,arith.CmpiOp(digit,number(body,nibble),"eq"))
                branch(body,match,lambda nested,n=nibble:string(nested,"0123456789abcdef"[n]))
            add(body,scf.YieldOp())
            add(where,scf.ForOp(number(where,0,builtin.IndexType()),number(where,count,builtin.IndexType()),
                number(where,1,builtin.IndexType()),[],Region(body)))

        def unsigned64(where, value):
            # Fixed twenty-digit domain, modular unsigned arithmetic including
            # 2**64-1; no float conversion, hidden allocation or signed cast.
            body = Block(arg_types=[builtin.IndexType(),builtin.i64,builtin.i1])
            _,divisor,started = body.args
            digit = add(body,arith.DivUIOp(value,divisor))
            digit = add(body,arith.RemUIOp(digit,number(body,10)))
            nonzero = add(body,arith.CmpiOp(digit,number(body,0),"ne"))
            last = add(body,arith.CmpiOp(divisor,number(body,1),"eq"))
            active = add(body,arith.OrIOp(started,nonzero))
            active = add(body,arith.OrIOp(active,last))
            branch(body,active,lambda nested:show(nested,digit))
            next_divisor = add(body,arith.DivUIOp(divisor,number(body,10)))
            add(body,scf.YieldOp(next_divisor,active))
            add(where,scf.ForOp(number(where,0,builtin.IndexType()),number(where,20,builtin.IndexType()),
                number(where,1,builtin.IndexType()),[number(where,10**19),number(where,0,builtin.i1)],Region(body)))

        def event(where, value, coordinates, shape, operand):
            header(where,coordinates,shape,operand)
            if report["hex"]:
                hex_word(where,value,value.type.bitwidth)
            elif isinstance(value.type,builtin.AnyFloat):
                if value.type in (builtin.f16,builtin.bf16):
                    value = add(where,arith.ExtFOp(value,builtin.f32))
                if value.type != builtin.f64:
                    value = add(where,arith.ExtFOp(value,builtin.f64))
                show(where,value)
            elif value.type == builtin.i64 and not report["is_signed"][operand]:
                unsigned64(where,value)
            else:
                show(where,integer64(where,value,report["is_signed"][operand]))
            string(where,"\n")

        for operand,value in enumerate(op.operands):
            shape = report["operand_shapes"][operand]
            def lanes(where, coordinates, axis):
                if axis == len(shape):
                    item = add(where,tensor.ExtractOp(value,coordinates,value.type.element_type)) if isinstance(value.type,builtin.TensorType) else value
                    event(where,item,coordinates,shape,operand)
                    return
                body = Block(arg_types=[builtin.IndexType()])
                lanes(body,[*coordinates,body.args[0]],axis+1)
                add(body,scf.YieldOp())
                add(where,scf.ForOp(number(where,0,builtin.IndexType()),number(where,shape[axis],builtin.IndexType()),
                    number(where,1,builtin.IndexType()),[],Region(body)))
            lanes(block,[],0)
        if not op.operands:
            header(block,None,[])
            string(block,"\n")
        operations = list(block.ops)
        for operation in operations:
            block.detach_op(operation)
        Rewriter.replace_op(op,operations)
        reports.append(report)
    return reports
