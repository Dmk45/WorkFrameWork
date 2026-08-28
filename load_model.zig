const std = @import("std");
const layers = @import("src/layers.zig");
const mloader_load = @import("src/mloader/load.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load the model
    const load_path = "test_model.sig";
    std.debug.print("Loading model from {s}...\n", .{load_path});
    var loaded_nn = try mloader_load.loadModel(allocator, load_path);
    defer loaded_nn.deinit();
    std.debug.print("Model loaded successfully with {d} layers\n", .{loaded_nn.num_layers()});

    // Print some info about the loaded model
    for (loaded_nn.layers.items, 0..) |layer, i| {
        switch (layer.layer_type) {
            .linear => {
                const linear = @as(*layers.LinearLayer, @ptrCast(@alignCast(layer.type)));
                std.debug.print("Layer {d}: Linear (input_size={d}, output_size={d})\n", .{i, linear.input_size, linear.output_size});
                std.debug.print("  Weights shape: ", .{});
                if (linear.weights.shape) |shape| {
                    for (shape.items) |dim| std.debug.print("{d} ", .{dim});
                    std.debug.print("\n", .{});
                }
            },
            .conv1d => {
                const conv1d = @as(*layers.Conv1DLayer, @ptrCast(@alignCast(layer.type)));
                std.debug.print("Layer {d}: Conv1D (in_channels={d}, out_channels={d}, kernel_size={d})\n", .{i, conv1d.in_channels, conv1d.out_channels, conv1d.kernel_size});
            },
            .conv2d => {
                const conv2d = @as(*layers.Conv2DLayer, @ptrCast(@alignCast(layer.type)));
                std.debug.print("Layer {d}: Conv2D (in_channels={d}, out_channels={d}, kernel_h={d}, kernel_w={d})\n", .{i, conv2d.in_channels, conv2d.out_channels, conv2d.kernel_h, conv2d.kernel_w});
            },
            .conv3d => {
                const conv3d = @as(*layers.Conv3DLayer, @ptrCast(@alignCast(layer.type)));
                std.debug.print("Layer {d}: Conv3D (in_channels={d}, out_channels={d}, kernel_d={d}, kernel_h={d}, kernel_w={d})\n", .{i, conv3d.in_channels, conv3d.out_channels, conv3d.kernel_d, conv3d.kernel_h, conv3d.kernel_w});
            },
            .lstm => {
                const lstm = @as(*layers.LSTMCell, @ptrCast(@alignCast(layer.type)));
                std.debug.print("Layer {d}: LSTM (input_size={d}, hidden_size={d})\n", .{i, lstm.input_size, lstm.hidden_size});
            },
            .gru => {
                const gru = @as(*layers.GRUCell, @ptrCast(@alignCast(layer.type)));
                std.debug.print("Layer {d}: GRU (input_size={d}, hidden_size={d})\n", .{i, gru.input_size, gru.hidden_size});
            },
        }
    }
}
