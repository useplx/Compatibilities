# Compatibilities

Compatibility packages for MyeLiN and SYNaPSe. Each directory represents one
target or target family. A package may provide operation selection, lowering,
backend compilation and a runtime driver without requiring the application to
import those parts separately.

The facets remain separate inside the package because they run at different
stages. For example, QNN contains:

```text
QNN
├── Lowering: prepares the selected StableHLO paths for QNN
├── Backend: compiles the QNN representation into a context binary
└── Driver: loads that context and dispatches it through QAIRT
```

The Driver accepts the binary format produced by the Backend but does not
depend on Lowering or compiler internals. Codec layout adaptations remain in
the quantization packages.

## Use

Import only the target package selected by the application's policies:

```toml
[dependencies]
adreno-643 = { path = "../Compatibilities/Adreno643" }
```

`Adreno643` imports the portable Vulkan package and adds only the quirks
validated for that device. CUDA and generic Vulkan applications import their
own packages, so unrelated strategies never enter the compiler process.

QNN applications import a single package:

```toml
[dependencies]
qnn = { path = "../Compatibilities/QNN" }
```

The QNN package requires a separately obtained Qualcomm AI Runtime SDK through
`QAIRT_ROOT`. The SDK, firmware, host sysroot, and proprietary runtime binaries
are not distributed by this repository. Resolving the package graph without a
target device is not hardware validation.

## Adreno 643 Projections

The Adreno 643 MatMul compatibility recognizes a static quantized-weight
projection from its `Transpose(Const)` producer. It does not apply to generic
MatMul operations such as the contractions inside Attention. Batched prefill
uses 256 output rows per tile. Ordinary single-token projections remain whole,
because splitting them increases submission cost, while vocabulary-wide
single-token projections use 512 output rows and 512 contracted elements per
segment. The compiler places each result directly in its output slice. The
tiled contractions preserve the codec's fused decode and FP32 accumulation.

Very wide single-token projections use the same tiled contraction inside a
compact loop plus a static tail. This avoids emitting hundreds of independent
dispatches for a vocabulary projection while keeping the per-workgroup shape
that is stable on the device.

Reductions are capped at 512 elements per semantic segment. This matches four
128-lane subgroups on the target and prevents large reductions from producing
pathological Vulkan workloads without forcing the model to encode a device
limit.

Output tiling increases compile time and VMFB size. It is not a universal
optimization and is not selected for CUDA, generic Adreno targets, other Vulkan
targets, dynamic input shapes or non-weight MatMul operations. Other Adreno
devices require their own validation before sharing these quirks.

The Q6A deployment additionally needs device capabilities matching its driver.
In particular, large weight bindings must respect the storage descriptor size
limit or use supported buffer device addresses. That deployment configuration
is separate from the mathematical MatMul compatibility.

## Vulkan Attention

The Vulkan package expands semantic Attention into equivalent MatMul, mask,
softmax and MatMul operations. The current upstream IREE SPIR-V pipeline cannot
legalize the temporary workgroup results produced by `LinalgExt::AttentionOp`
for this target. The model still declares Attention semantically; only the
imported Vulkan compatibility selects the portable execution form.

## Validation

`syn_compatibility_selection` checks target and shape selection.
`mln_tiled_matmul_test` checks graph structure and can compile diagnostic
projections. `Quantizations/tests/tiled_matmul_runtime_test.py` compares their
outputs with a decoded NumPy reference. Its explicit rounding option separates
final FP16 conversion behavior from contraction errors.

The batched 6144-row projection was also compared against the untiled path with
the same Q4K weight and input on Adreno 643. Both paths produced byte-identical
FP16 output.

Wide single-token projections are emitted as static output tiles with a
cooperative contraction. Keeping the reduction semantic lets IREE distribute
the contraction across the Adreno subgroup instead of assigning an entire
row-major dot product to each invocation. With the Qwen3.5 LM head this reduced
the isolated median from 1.043 seconds to 0.561 seconds. The changed reduction
order was checked with real Q4K weights and a random FP16 input: the maximum
absolute difference was one FP16 ULP and the mean absolute difference was
1.06e-7.

Passing an isolated projection is not sufficient to certify a complete model
or a different Vulkan driver.

## License

PLeX compatibility source code is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Vendor SDKs and drivers retain
their own terms and are not covered or redistributed by this license.
