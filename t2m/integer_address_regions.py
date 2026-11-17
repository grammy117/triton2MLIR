"""Pair logical bases with byte state across original structured region edges.

Reuse the edge conventions and monotone worklist from pointer_control.py and
pointer_select.py (Meta/shared 17f0c234 base/offset traversal). While input and
output interfaces are independent. No source loop, branch or effect is moved.
"""

from collections import deque
from itertools import product

from xdsl.dialects import builtin
from xdsl.ir import BlockArgument, OpResult, SSAValues
from xdsl.utils.exceptions import ParseError

from .diagnostics import ConversionError
from .pointer_addresses import SHAPES, Address, _factor, _like, _scalar
from .pointer_views import element


def analyze(module, source_ids):
    from .shared_tensor import opcode

    operations = tuple(module.walk())
    if (not any(opcode(op) == "tt.ptr_to_int" for op in operations)
            or not any(opcode(op) in ("scf.for", "scf.while", "scf.if") for op in operations)):
        return None
    values = [value for op in operations for value in (
        *op.results, *(arg for region in op.regions for block in region.blocks for arg in block.args))
        if isinstance(_scalar(value.type), builtin.IntegerType) and 1 <= _scalar(value.type).bitwidth <= 64]
    if len(values) > 65536:
        raise ConversionError("integer address region analysis exceeds its integer SSA budget")
    roles = {value: set() for value in values}
    dependencies, rules = {}, {}

    def refuse(op, message):
        raise ConversionError(message, opcode=opcode(op), source_id=next(iter(source_ids(op.location)), None))

    def block(op, number):
        if len(op.regions) <= number or len(op.regions[number].blocks) != 1:
            refuse(op, "integer address state needs single-block structured regions")
        return op.regions[number].block

    def end(op, number, name):
        terminal = block(op, number).last_op
        if terminal is None or opcode(terminal) != name:
            refuse(op, "integer address state has an unexpected region terminator")
        return terminal

    for value in values:
        if isinstance(value, BlockArgument):
            op, index = value.block.parent_op(), value.index
            name = opcode(op)
            if name == "scf.for" and index:
                dependencies[value] = (op.operands[index + 2], end(op, 0, "scf.yield").operands[index - 1])
                rules[value] = "join"
            elif name == "scf.while":
                dependencies[value] = ((op.operands[index], end(op, 1, "scf.yield").operands[index])
                    if value.block is block(op, 0) else (end(op, 0, "scf.condition").operands[index + 1],))
                rules[value] = "join"
            else:
                dependencies[value], rules[value] = (), "numeric"
            continue
        op, index = value.owner, value.index
        name = opcode(op)
        if name == "tt.ptr_to_int":
            dependencies[value], rules[value] = (), "address"
        elif name in ("arith.addi", "arith.subi"):
            dependencies[value], rules[value] = tuple(op.operands), name
        elif name == "arith.select":
            dependencies[value], rules[value] = tuple(op.operands[1:]), "join"
        elif name in SHAPES or name == "arith.bitcast" and op.operands[0].type == value.type:
            dependencies[value], rules[value] = tuple(op.operands), "join"
        elif name == "scf.for":
            dependencies[value], rules[value] = (op.operands[index + 3], end(op, 0, "scf.yield").operands[index]), "join"
        elif name == "scf.if":
            dependencies[value], rules[value] = tuple(end(op, r, "scf.yield").operands[index] for r in (0, 1)), "join"
        elif name == "scf.while":
            dependencies[value], rules[value] = (end(op, 0, "scf.condition").operands[index + 1],), "join"
        else:
            dependencies[value], rules[value] = (), "numeric"
    users = {value: [] for value in values}
    for value, incoming in dependencies.items():
        for source in incoming:
            if source in users:
                users[source].append(value)

    def worklist(update, graph=users):
        pending, queued = deque(values), set(values)
        while pending:
            value = pending.popleft()
            queued.remove(value)
            if update(value):
                for user in graph[value]:
                    if user not in queued:
                        pending.append(user)
                        queued.add(user)

    def classify(value):
        rule = rules[value]
        incoming = [roles.get(source, {"N"}) for source in dependencies[value]]
        if rule in ("address", "numeric"):
            found = {"A" if rule == "address" else "N"}
        elif rule == "join":
            found = set().union(*incoming)
        else:
            found = set()
            for left, right in product(*incoming):
                if "X" in (left, right):
                    found.add("X")
                elif rule == "arith.addi":
                    found.add("X" if left == right == "A" else "A" if "A" in (left, right) else "N")
                else:
                    found.add("N" if left == right else "A" if left == "A" else "X")
        previous = len(roles[value])
        roles[value].update(found)
        return len(roles[value]) != previous

    worklist(classify)
    addresses = {value for value in values if roles[value] == {"A"}}
    controls = {op for op in operations if opcode(op) in ("scf.for", "scf.while", "scf.if")
                and any(value in addresses or "A" in roles.get(value, ()) for value in (
                    *op.operands, *op.results,
                    *(arg for region in op.regions for body in region.blocks for arg in body.args)))}
    if not controls:
        return None
    if len(addresses) > 8192:
        raise ConversionError("integer address region analysis exceeds its address SSA budget")
    for value in values:
        if len(roles[value]) > 1 or roles[value] == {"X"}:
            op = value.block.parent_op() if isinstance(value, BlockArgument) else value.owner
            refuse(op, "integer address regions require closed address-only state, separate from ordinary integers")
    origins = {value: {value.owner} if rules[value] == "address" else set() for value in addresses}
    root_members = sum(map(len, origins.values()))

    def roots(value):
        nonlocal root_members
        if value not in addresses or rules[value] == "address":
            return False
        incoming = set().union(*(origins.get(source, set()) for source in dependencies[value]))
        old = len(origins[value])
        origins[value].update(incoming)
        root_members += len(origins[value]) - old
        if root_members > 262144 or len(origins[value]) > 4096:
            raise ConversionError("integer address region analysis exceeds its root propagation budget")
        return len(origins[value]) != old

    worklist(roots)
    factors = dict.fromkeys(values, 3)
    factor_users = {value: [] for value in values}
    for value in values:
        incoming = dependencies[value] if isinstance(value, BlockArgument) or rules[value] == "join" else value.owner.operands
        for source in incoming:
            if source in factor_users:
                factor_users[source].append(value)

    def alignment(value):
        if value in addresses:
            if rules[value] == "address":
                return False
            found = min((factors.get(source, 0) for source in dependencies[value]), default=0)
        elif rules[value] == "join":
            found = min((factors.get(source, 0) for source in dependencies[value]), default=0)
        else:
            # A numeric difference of two absolute addresses gets no bit fact
            # until its actual common base is proved. Ordinary arithmetic can
            # use only ordinary numeric facts, including cyclic numeric phis.
            found = _factor(value, lambda source: factors.get(source, 0) if source not in addresses else 0)
        old = factors[value]
        factors[value] = min(old, found)
        return old != factors[value]

    worklist(alignment, factor_users)
    return RegionAddresses(operations, addresses, origins, factors, controls, source_ids)


class RegionAddresses:
    def __init__(self, operations, addresses, origins, factors, controls, source_ids):
        self.operations, self.addresses, self.origins, self.factors = operations, addresses, origins, factors
        self.controls, self.source_ids = controls, source_ids
        self.operands = {op: tuple(op.operands) for op in operations}
        self.results = {op: tuple(op.results) for op in operations}
        self.args = {body: tuple(body.args) for op in operations for region in op.regions for body in region.blocks}
        order = {op: i for i, op in enumerate(operations)}
        self.seeds = {value: sorted(origins[value], key=order.__getitem__) for value in addresses}
        self.pointers, self.pairs, self.links, self.aliases = {}, {}, {}, {}
        self.field_controls = {}
        self.reports = []

    def prepare(self):
        from .shared_tensor import opcode

        operations, controls, addresses, source_ids = self.operations, self.controls, self.addresses, self.source_ids
        for op in operations:
            if op not in controls:
                continue
            name = opcode(op)
            for value in self.results[op]:
                if value in addresses:
                    pointer = OpResult(self.pointer_type(value), op, len(op.results))
                    op.results = SSAValues((*op.results, pointer))
                    self.register(value, pointer)
            for region in op.regions:
                for body in region.blocks:
                    for value in self.args[body]:
                        if value in addresses:
                            self.register(value, body.insert_arg(self.pointer_type(value), len(body.args)))
            if name == "scf.for":
                for result, argument in zip(self.results[op], self.args[op.regions[0].block][1:], strict=True):
                    if result in addresses:
                        self.links[self.pointers[result]] = [self.pointers[argument]]
            elif name == "scf.while":
                for result, argument in zip(self.results[op], self.args[op.regions[1].block], strict=True):
                    if result in addresses:
                        self.links[self.pointers[argument]] = [self.pointers[result]]
            self.reports.append({"kind": "integer-address-region-state", "source_ids": source_ids(op.location),
                                 "control": name, "representation": "paired-logical-pointer-base-and-i64-byte-offset",
                                 "arithmetic": "modulo-2^64; original-region-edges-and-effects-retained",
                                 "physical_address": "unobserved", "public_abi": "unchanged"})
        return self

    def pointer_type(self, value):
        if not self.seeds[value]:
            raise ConversionError("integer address state has no established source pointer")
        pointer = self.seeds[value][0].operands[0]
        try:
            element(pointer.type)
        except (ValueError, ParseError) as error:
            seed = self.seeds[value][0]
            raise ConversionError(str(error), opcode="tt.ptr_to_int",
                                  source_id=next(iter(self.source_ids(seed.location)), None)) from error
        return _like(value.type, _scalar(pointer.type))

    def register(self, value, pointer):
        self.pointers[value] = pointer
        self.field_controls[pointer] = value.block.parent_op() if isinstance(value, BlockArgument) else value.owner
        seed = self.seeds[value][0]
        ids = frozenset(identity for op in self.seeds[value] for identity in self.source_ids(op.location))
        self.pairs[value] = Address(pointer, value, ids, seed)

    def handle(self, op, states, emitted):
        from .shared_tensor import opcode

        name = opcode(op)
        if op not in self.controls and name not in ("scf.yield", "scf.condition"):
            return False
        original = self.operands[op]
        targets = None
        if name == "scf.for" and op in self.controls:
            targets = self.args[op.regions[0].block][1:]
            prefix = 3
        elif name == "scf.while" and op in self.controls:
            targets, prefix = self.args[op.regions[0].block], 0
        elif name in ("scf.yield", "scf.condition"):
            parent = op.parent_op()
            if parent not in self.controls:
                return False
            control = opcode(parent)
            if control == "scf.for":
                targets, prefix = self.args[parent.regions[0].block][1:], 0
            elif control == "scf.if":
                targets, prefix = self.results[parent], 0
            elif control == "scf.while":
                targets, prefix = (self.results[parent], 1) if name == "scf.condition" else (
                    self.args[parent.regions[0].block], 0)
        elif name == "scf.if" and op in self.controls:
            targets, prefix = (), len(original)
        if targets is None:
            return False
        if any(value in self.addresses for value in original[:prefix]):
            raise ConversionError("physical pointer integers cannot control region bounds or conditions",
                                  opcode=name, source_id=next(iter(self.source_ids(op.location)), None))
        operands, pointers = list(original), []
        for index, target in enumerate(targets):
            if target not in self.addresses:
                continue
            source = original[index + prefix]
            if source not in states:
                raise ConversionError("integer address edge has no closed incoming pair", opcode=name,
                                      source_id=next(iter(self.source_ids(op.location)), None))
            address = states[source]
            pointer = address.pointer
            type_ = self.pointer_type(target)
            if pointer.type != type_:
                element(pointer.type)
                cast = builtin.UnregisteredOp.with_name("tt.bitcast").create(operands=[pointer], result_types=[type_])
                emitted.append(cast)
                pointer = cast.results[0]
            operands[index + prefix] = address.displacement
            pointers.append(pointer)
            self.links.setdefault(self.pointers[target], []).append(pointer)
        op.operands = [*operands, *pointers]
        return True

    def resolve(self, value):
        seen = set()
        while value in self.aliases and value not in seen:
            seen.add(value)
            value = self.aliases[value]
        return value

    def fold_bases(self):
        def visible(value, operation):
            block = value.block if isinstance(value, BlockArgument) else value.owner.parent
            if block is None:
                return False
            current = operation
            while current is not None and current.parent is not block:
                current = current.parent_op()
            if current is None:
                return False
            return isinstance(value, BlockArgument) or value.owner is not current and value.owner.is_before_in_block(current)

        # A preceding branch result is a stable anchor for a later loop even
        # when that branch chose among several roots. Stop at each control's
        # external SSA boundary; nested controls fold before their parents.
        for control in reversed(self.operations):
            if control not in self.controls:
                continue
            roots = {value: set() for value, owner in self.field_controls.items() if owner is control}
            users = {value: [] for value in roots}
            for target, candidates in roots.items():
                for original in self.links.get(target, ()):
                    value = self.resolve(original)
                    if value in roots:
                        users[value].append(target)
                    else:
                        candidates.add(value)
            members = sum(map(len, roots.values()))
            pending = deque(value for value in roots if roots[value])
            while pending:
                value = pending.popleft()
                for user in users[value]:
                    old = len(roots[user])
                    roots[user].update(roots[value])
                    members += len(roots[user]) - old
                    if members > 262144 or len(roots[user]) > 4096:
                        raise ConversionError("integer address base folding exceeds its root propagation budget")
                    if len(roots[user]) != old:
                        pending.append(user)
            for value, candidates in roots.items():
                if len(candidates) == 1:
                    candidate = next(iter(candidates))
                    # Deferred byte differences have not registered all uses
                    # yet. The candidate must dominate the carrier itself.
                    if visible(candidate, control) and all(visible(candidate, use.operation) for use in value.uses):
                        self.aliases[value] = candidate
                        value.replace_all_uses_with(candidate)
