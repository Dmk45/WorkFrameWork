const std = @import("std");
const layers = @import("src/layers.zig");
const mloader_save = @import("src/mloader/save.zig");
const mloader_load = @import("src/mloader/load.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Creating test model...\n", .{});

    // Create a simple neural network
    var nn = try layers.NeuralNetwork.init(allocator);
    defer nn.deinit();

    // Add some linear layers
    const l1 = try layers.LinearLayer.init(allocator, 10, 20, "relu", null, null);
    try nn.add(l1);

    const l2 = try layers.LinearLayer.init(allocator, 20, 5, "sigmoid", null, null);
    try nn.add(l2);

    std.debug.print("Model created with {d} layers\n", .{nn.num_layers()});

    // Save the model
    const save_path = "test_model.sig";
    std.debug.print("Saving model to {s}...\n", .{save_path});
    try mloader_save.saveModel(allocator, &nn, save_path);
    std.debug.print("Model saved successfully\n", .{});

    // Load the model
    std.debug.print("Loading model from {s}...\n", .{save_path});
    var loaded_nn = try mloader_load.loadModel(allocator, save_path);
    defer loaded_nn.deinit();
    std.debug.print("Model loaded successfully with {d} layers\n", .{loaded_nn.num_layers()});

    // Verify structure
    if (loaded_nn.num_layers() != nn.num_layers()) {
        std.debug.print("ERROR: Layer count mismatch\n", .{});
        return error.TestFailed;
    }

    // Compare first layer weights
    const orig_l1 = @as(*layers.LinearLayer, @ptrCast(@alignCast(nn.layers.items[0].type)));
    const loaded_l1 = @as(*layers.LinearLayer, @ptrCast(@alignCast(loaded_nn.layers.items[0].type)));

    if (orig_l1.weights.values.items.len != loaded_l1.weights.values.items.len) {
        std.debug.print("ERROR: Weights size mismatch\n", .{});
        return error.TestFailed;
    }

    var weights_match = true;
    for (orig_l1.weights.values.items, 0..) |val, i| {
        if (@abs(val - loaded_l1.weights.values.items[i]) > 0.0001) {
            weights_match = false;
            break;
        }
    }

    if (!weights_match) {
        std.debug.print("ERROR: Weights values mismatch\n", .{});
        return error.TestFailed;
    }

    std.debug.print("SUCCESS: Model persistence test passed!\n", .{});

    // Clean up test file
    std.fs.cwd().deleteFile(save_path) catch {};
}
