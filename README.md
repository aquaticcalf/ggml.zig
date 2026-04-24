# ggml.zig

A Zig wrapper for the [GGML](https://github.com/ggerganov/ggml) library - a tensor library for machine learning.

## Features

- Zig-friendly API wrapping GGML C functions
- Automatic GGML cloning from GitHub during build
- Static library compilation with Zig
- Support for tensor operations, backends, and graph computation
- GGUF model format support

## Installation

Add this dependency to your Zig project:

```bash
zig fetch --save https://github.com/aquaticcalf/ggml.zig/archive/refs/heads/master.tar.gz
```

This will add `ggml` to your `build.zig.zon` dependencies.

## Building

```bash
zig build
```

This will automatically clone GGML from GitHub and build it.

## Testing

```bash
zig build test
```

## Usage

In your `build.zig`, add the dependency:

```zig
const ggml_dep = b.dependency("ggml", .{
    .target = target,
    .optimize = optimize,
});

// For executables:
exe.root_module.addImport("ggml", ggml_dep.module("ggml"));

// For libraries:
lib.root_module.addImport("ggml", ggml_dep.module("ggml"));
```

Then in your Zig code:

```zig
const ggml = @import("ggml");

// Initialize context
const params = ggml.InitParams{
    .mem_size = 16 * 1024 * 1024,
    .mem_buffer = null,
    .no_alloc = false,
};
const ctx = ggml.init(params).?;

// Create tensors
const tensor = ggml.newTensor1D(ctx, ggml.GGML_TYPE_F32, 10);

// Access data
const data = ggml.getF32Data(tensor.?);
data[0] = 1.0;

// Perform operations
const result = ggml.add(ctx, a, b);
```

## Example

See `examples/basic.zig` for a complete example.

Build and run the example:

```bash
zig build-exe examples/basic.zig -I vendor/ggml/include -I vendor/ggml/src -I vendor/ggml/src/ggml-cpu -lc++ -L zig-out/lib -lggml
```

## API Coverage

- Context management (`init`, `free`)
- Tensor creation (1D, 2D, 3D, 4D)
- Tensor operations (add, sub, mul, div, pow, etc.)
- Unary operations (abs, neg, sqrt, sin, cos, etc.)
- Matrix operations (mul_mat, transpose, etc.)
- Reduction operations (sum, mean, etc.)
- Normalization (rms_norm, norm)
- RoPE (rope)
- Softmax and diag mask
- Graph computation
- Backend API
- Scheduler
- GGUF support

## License

MIT (same as GGML)
