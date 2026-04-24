const std = @import("std");
const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-backend.h");
});

pub const InitParams = c.ggml_init_params;
pub const Type = c.ggml_type;

// GGML types
pub const GGML_TYPE_F32 = c.GGML_TYPE_F32;
pub const GGML_TYPE_F16 = c.GGML_TYPE_F16;
pub const GGML_TYPE_Q4_0 = c.GGML_TYPE_Q4_0;
pub const GGML_TYPE_Q4_1 = c.GGML_TYPE_Q4_1;
pub const GGML_TYPE_Q5_0 = c.GGML_TYPE_Q5_0;
pub const GGML_TYPE_Q5_1 = c.GGML_TYPE_Q5_1;
pub const GGML_TYPE_Q8_0 = c.GGML_TYPE_Q8_0;
pub const GGML_TYPE_Q8_1 = c.GGML_TYPE_Q8_1;
pub const GGML_TYPE_Q2_K = c.GGML_TYPE_Q2_K;
pub const GGML_TYPE_Q3_K = c.GGML_TYPE_Q3_K;
pub const GGML_TYPE_Q4_K = c.GGML_TYPE_Q4_K;
pub const GGML_TYPE_Q5_K = c.GGML_TYPE_Q5_K;
pub const GGML_TYPE_Q6_K = c.GGML_TYPE_Q6_K;
pub const GGML_TYPE_I8 = c.GGML_TYPE_I8;
pub const GGML_TYPE_I16 = c.GGML_TYPE_I16;
pub const GGML_TYPE_I32 = c.GGML_TYPE_I32;

// Opaque types
pub const Tensor = *c.struct_ggml_tensor;
pub const Context = *c.struct_ggml_context;
pub const CGraph = *c.struct_ggml_cgraph;
pub const Backend = *anyopaque;
pub const BackendBuffer = *anyopaque;
pub const BackendBufferType = *anyopaque;
pub const Scheduler = *anyopaque;
pub const GGUFContext = *anyopaque;

// Context management
pub fn init(params: InitParams) ?Context {
    return @ptrCast(c.ggml_init(params));
}

pub fn free(ctx: Context) void {
    c.ggml_free(ctx);
}

// Tensor creation
pub fn newTensor1D(ctx: Context, typ: Type, ne0: i64) ?Tensor {
    return @ptrCast(c.ggml_new_tensor_1d(ctx, typ, ne0));
}

pub fn newTensor2D(ctx: Context, typ: Type, ne0: i64, ne1: i64) ?Tensor {
    return @ptrCast(c.ggml_new_tensor_2d(ctx, typ, ne0, ne1));
}

pub fn newTensor3D(ctx: Context, typ: Type, ne0: i64, ne1: i64, ne2: i64) ?Tensor {
    return @ptrCast(c.ggml_new_tensor_3d(ctx, typ, ne0, ne1, ne2));
}

pub fn newTensor4D(ctx: Context, typ: Type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) ?Tensor {
    return @ptrCast(c.ggml_new_tensor_4d(ctx, typ, ne0, ne1, ne2, ne3));
}

// Tensor info
pub fn nelements(tensor: Tensor) i64 {
    return c.ggml_nelements(tensor);
}

pub fn nbytes(tensor: Tensor) usize {
    return c.ggml_nbytes(tensor);
}

pub fn getType(tensor: Tensor) Type {
    return c.ggml_type(tensor);
}

pub fn getN(ctx: Context, tensor: Tensor, i: i32) i64 {
    return c.ggml_get_n(@ptrCast(ctx), tensor, i);
}

// Tensor data access
pub fn getData(tensor: Tensor) [*]u8 {
    return @ptrCast(@alignCast(c.ggml_get_data(tensor)));
}

pub fn getF32Data(tensor: Tensor) [*]f32 {
    return @ptrCast(@alignCast(c.ggml_get_data(tensor)));
}

pub fn dupTensor(ctx: Context, tensor: Tensor) ?Tensor {
    return @ptrCast(c.ggml_dup_tensor(ctx, tensor));
}

pub fn setZero(tensor: Tensor) void {
    c.ggml_set_zero(tensor);
}

pub fn set(ctx: Context, tensor: Tensor, value: f32) void {
    c.ggml_set(@ptrCast(ctx), tensor, value);
}

// Unary operations
pub fn abs(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_abs(@ptrCast(ctx), a));
}

pub fn neg(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_neg(@ptrCast(ctx), a));
}

pub fn step(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_step(@ptrCast(ctx), a));
}

pub fn rectify(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_relu(@ptrCast(ctx), a));
}

pub fn gelu(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_gelu(@ptrCast(ctx), a));
}

pub fn sqr(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_sqr(@ptrCast(ctx), a));
}

pub fn sqrt(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_sqrt(@ptrCast(ctx), a));
}

pub fn sin(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_sin(@ptrCast(ctx), a));
}

pub fn cos(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_cos(@ptrCast(ctx), a));
}

// Binary operations
pub fn add(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_add(@ptrCast(ctx), a, b));
}

pub fn sub(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_sub(@ptrCast(ctx), a, b));
}

pub fn mul(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_mul(@ptrCast(ctx), a, b));
}

pub fn div(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_div(@ptrCast(ctx), a, b));
}

pub fn pow(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_pow(@ptrCast(ctx), a, b));
}

// Matrix operations
pub fn mulMat(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_mul_mat(@ptrCast(ctx), a, b));
}

pub fn transpose(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_transpose(@ptrCast(ctx), a));
}

// Reduction operations
pub fn sum(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_sum(@ptrCast(ctx), a));
}

pub fn sumRows(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_sum_rows(@ptrCast(ctx), a));
}

pub fn mean(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_mean(@ptrCast(ctx), a));
}

// Tensor manipulation
pub fn contiguous(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_cont(@ptrCast(ctx), a));
}

pub fn permute(ctx: Context, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) ?Tensor {
    return @ptrCast(c.ggml_permute(@ptrCast(ctx), a, ne0, ne1, ne2, ne3));
}

pub fn reshape(ctx: Context, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) ?Tensor {
    return @ptrCast(c.ggml_reshape_4d(@ptrCast(ctx), a, ne0, ne1, ne2, ne3));
}

pub fn view1D(ctx: Context, a: Tensor, ne0: i64, offset: usize) ?Tensor {
    return @ptrCast(c.ggml_view_1d(@ptrCast(ctx), a, ne0, offset));
}

// Scale and normalization
pub fn scale(ctx: Context, a: Tensor, b: Tensor) ?Tensor {
    return @ptrCast(c.ggml_scale(@ptrCast(ctx), a, b));
}

pub fn rmsNorm(ctx: Context, a: Tensor, eps: f32) ?Tensor {
    return @ptrCast(c.ggml_rms_norm(@ptrCast(ctx), a, eps));
}

pub fn norm(ctx: Context, a: Tensor, eps: f32) ?Tensor {
    return @ptrCast(c.ggml_norm(@ptrCast(ctx), a, eps));
}

// RoPE
pub fn rope(ctx: Context, a: Tensor, n_dims: i32, mode: i32, n_ctx_orig: i32, n_ctx: i32, n_orig_y: i32, freq_base: f32, freq_scale: f32) ?Tensor {
    return @ptrCast(c.ggml_rope(@ptrCast(ctx), a, n_dims, mode, n_ctx_orig, n_ctx, n_orig_y, freq_base, freq_scale));
}

// Softmax
pub fn softmax(ctx: Context, a: Tensor) ?Tensor {
    return @ptrCast(c.ggml_soft_max(@ptrCast(ctx), a));
}

// Diag mask
pub fn diagMask(ctx: Context, a: Tensor, n_past: i32) ?Tensor {
    return @ptrCast(c.ggml_diag_mask_inf(@ptrCast(ctx), a, n_past));
}

// Threading
pub fn useThreading(ctx: Context, n_threads: c_int) void {
    c.ggml_threading_set_n_threads(ctx, n_threads);
}

// Graph computation
pub fn graphCompute(ctx: Context, gf: CGraph) void {
    c.ggml_graph_compute(ctx, gf);
}

pub fn graphReset(ctx: Context, gf: CGraph) void {
    c.ggml_graph_reset(ctx, gf);
}

pub fn newGraph(ctx: Context) ?CGraph {
    return @ptrCast(c.ggml_new_graph(@ptrCast(ctx)));
}

// Backend API
pub fn backendInit() ?Backend {
    return @ptrCast(c.ggml_backend_init());
}

pub fn backendFree(backend: Backend) void {
    c.ggml_backend_free(@ptrCast(backend));
}

pub fn backendName(backend: Backend) [*:0]const u8 {
    return c.ggml_backend_name(@ptrCast(backend));
}

pub fn backendAllocTensor(backend: Backend, tensor: Tensor) void {
    c.ggml_backend_alloc_tensor(@ptrCast(backend), tensor);
}

pub fn backendAllocCtx(backend: Backend, ctx: Context) void {
    c.ggml_backend_alloc_ctx_tensors_from_buft(@ptrCast(backend), @ptrCast(ctx));
}

// GGUF support
pub fn ggufInitEmpty() ?GGUFContext {
    return @ptrCast(c.gguf_init_empty());
}

pub fn ggufFree(ctx: GGUFContext) void {
    c.gguf_free(@ptrCast(ctx));
}

// Threading
pub fn getNumThreads() c_int {
    return c.ggml_threading_get_n_threads();
}

test "ggml init and free" {
    const params = InitParams{
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    };
    const ctx = init(params);
    try std.testing.expect(ctx != null);
    free(ctx.?);
}

test "ggml tensor creation" {
    const params = InitParams{
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    };
    const ctx = init(params).?;
    defer free(ctx);

    const tensor = newTensor1D(ctx, GGML_TYPE_F32, 10);
    try std.testing.expect(tensor != null);
    try std.testing.expect(nelements(tensor.?) == 10);
}

test "ggml tensor data access" {
    const params = InitParams{
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    };
    const ctx = init(params).?;
    defer free(ctx);

    const tensor = newTensor1D(ctx, GGML_TYPE_F32, 5);
    try std.testing.expect(tensor != null);

    const data = getF32Data(tensor.?);
    data[0] = 1.0;
    data[1] = 2.0;
    try std.testing.expect(@abs(data[0] - 1.0) < 0.001);
    try std.testing.expect(@abs(data[1] - 2.0) < 0.001);
}

test "ggml tensor operations" {
    const params = InitParams{
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    };
    const ctx = init(params).?;
    defer free(ctx);

    const a = newTensor1D(ctx, GGML_TYPE_F32, 3);
    const b = newTensor1D(ctx, GGML_TYPE_F32, 3);
    try std.testing.expect(a != null and b != null);

    const data_a = getF32Data(a.?);
    const data_b = getF32Data(b.?);
    data_a[0] = 1.0;
    data_a[1] = 2.0;
    data_a[2] = 3.0;
    data_b[0] = 4.0;
    data_b[1] = 5.0;
    data_b[2] = 6.0;

    const result = add(ctx, a.?, b.?);
    try std.testing.expect(result != null);
}
