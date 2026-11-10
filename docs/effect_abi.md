# Effect-only kernel ABI

`t2m.shared-tensor-abi.v6` explicitly selects `effect_only: true`. It permits
readonly tensor inputs, runtime scalars, static specializations and a logical
grid, including an empty argument list. The function has no tensor results.
Writable buffers cannot disappear into this profile: they require the usual
functional result ABI. Earlier schemas keep their existing serialization and
continue requiring writable results.

The converter retains the standard source computation and observable effects.
Scalar and dynamically shaped tensor assertions already execute through the
native JIT with a real void entry in all memory profiles. No result placeholder
or artificial allocation is introduced. Named dimensions and range declarations
retain their existing executable guards.

This ABI alone is not a complete capture package. Effect-only packages need an
explicit outcome reference and an effect-aware compiler intake. Numeric package
writers must continue using their tensor-result contract. The next package work
binds returned or source-assertion-failure outcomes, exact diagnostic streams,
source/payload bytes and actual execution evidence independently of compiler
inputs. Existing success-only golden files cannot represent an always-failing
source assertion.
