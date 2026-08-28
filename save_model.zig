const std = @import("std");
const layers = @import("src/layers.zig");
const mloader_save = @import("src/mloader/save.zig");

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
}
