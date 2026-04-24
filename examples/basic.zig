const std = @import("std");
const ggml = @import("ggml");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Initialize GGML context
    const params = ggml.InitParams{
        .mem_size = 16 * 1024 * 1024, // 16MB
        .mem_buffer = null,
        .no_alloc = false,
    };

    const ctx = ggml.init(params) orelse {
        std.debug.print("Failed to initialize GGML context\n", .{});
        return;
    };
    defer ggml.free(ctx);

    // Create tensors
    const a = ggml.newTensor1D(ctx, ggml.GGML_TYPE_F32, 3) orelse {
        std.debug.print("Failed to create tensor a\n", .{});
        return;
    };

    const b = ggml.newTensor1D(ctx, ggml.GGML_TYPE_F32, 3) orelse {
        std.debug.print("Failed to create tensor b\n", .{});
        return;
    };

    // Set data
    const data_a = ggml.getF32Data(a);
    const data_b = ggml.getF32Data(b);
    data_a[0] = 1.0;
    data_a[1] = 2.0;
    data_a[2] = 3.0;
    data_b[0] = 4.0;
    data_b[1] = 5.0;
    data_b[2] = 6.0;

    // Perform addition: result = a + b
    const result = ggml.add(ctx, a, b) orelse {
        std.debug.print("Failed to perform addition\n", .{});
        return;
    };

    const data_result = ggml.getF32Data(result);
    try stdout.print("Tensor a: [{d}, {d}, {d}]\n", .{ data_a[0], data_a[1], data_a[2] });
    try stdout.print("Tensor b: [{d}, {d}, {d}]\n", .{ data_b[0], data_b[1], data_b[2] });
    try stdout.print("Result a + b: [{d}, {d}, {d}]\n", .{ data_result[0], data_result[1], data_result[2] });
    try stdout.print("GGML Zig wrapper is working!\n", .{});
}
