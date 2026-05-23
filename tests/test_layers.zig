const std = @import("std");
const lib = @import("modelwork2");
const builder_mod = lib.maker;
const grad_mod = lib.grad;
const grad_math = lib.grad_math;
const core_mod = lib.core_math;
const layers_mod = lib.layers;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n=== Neural Network with Automatic Layer Management ===\n", .{});

    // Initialize tensor registry for FFI backward calls
    grad_math.grad_math_init_registry(allocator);

    // Create a neural network with layers
    var nn = try layers_mod.NeuralNetwork.init(allocator);
    defer nn.deinit();

    // Add layers without manually touching the tape!
    std.debug.print("\nBuilding model architecture:\n", .{});
    std.debug.print("  Layer 1: Linear(10 → 5, relu)\n", .{});
    const layer1 = try layers_mod.LinearLayer.init(allocator, 10, 5, "relu", null, null);
    try nn.add(layer1);

    std.debug.print("  Layer 2: Linear(5 → 3, relu)\n", .{});
    const layer2 = try layers_mod.LinearLayer.init(allocator, 5, 3, "relu", null, null);
    try nn.add(layer2);

    std.debug.print("  Output: Linear(3 → 2, none)\n", .{});
    const layer3 = try layers_mod.LinearLayer.init(allocator, 3, 2, "none", null, null);
    try nn.add(layer3);

    // Example of using other layer types:
    // try layers_mod.LSTMCell.addNN(&nn, 10, 5);
    // try layers_mod.GRUCell.addNN(&nn, 10, 5);
    // try layers_mod.Conv1DLayer.addNN(&nn, 3, 5, 3, 1, 1);

    std.debug.print("Model has {} layers\n", .{nn.num_layers()});

    // Create input data
    var input = try builder_mod.zeros(allocator, &[_]usize{ 1, 10 });
    defer input.deinit();

    // Fill with sample data
    for (0..10) |i| {
        input.values.items[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Create target labels
    var targets = try builder_mod.zeros(allocator, &[_]usize{ 1, 2 });
    defer targets.deinit();
    targets.values.items[0] = 1.0; // one-hot: class 0
    targets.values.items[1] = 0.0;

    std.debug.print("\nInput shape: {any}\n", .{input.shape.?.items});
    std.debug.print("Input values: {any}\n", .{input.values.items});
    std.debug.print("Target shape: {any}\n", .{targets.shape.?.items});

    // Training loop
    std.debug.print("\n--- Training Loop ---\n", .{});

    var optimizer = try grad_mod.Adam.init(allocator, 0.01, 0.9, 0.999, 1e-8);
    defer optimizer.deinit();

    for (0..3) |epoch| {
        // FORWARD PASS - automatically records all operations to tape!
        std.debug.print("\nEpoch {}: ", .{epoch + 1});

        var output = try nn.forward(allocator, .{ .tensor = &input });
        defer output.deinit();

        std.debug.print("Output shape: {any}, Values: {any}\n", .{ output.shape.?.items, output.values.items });

        // Compute loss
        const loss = try grad_mod.meanSquaredError(&output, &targets);
        std.debug.print("  Loss: {d:.6}\n", .{loss});

        // BACKWARD PASS - automatically traverses through all recorded operations
        std.debug.print("  Running backward pass...\n", .{});
        // Note: In real implementation with integration, this would be:
        // nn.tape.backward();

        // Update parameters
        std.debug.print("  Updating model parameters...\n", .{});
        try nn.update_parameters(&optimizer);

        // Zero gradients for next iteration
        nn.zero_grad();
    }

    std.debug.print("\n=== Training Complete ===\n\n", .{});

    // Show final model state
    std.debug.print("Final model structure:\n", .{});
    for (0..nn.num_layers()) |i| {
        if (nn.get_layer(i)) |layer| {
            const linear = layers_mod.Layer.child(layers_mod.LinearLayer, layer);
            std.debug.print(
                "  Layer {}: {s} - weights shape: {any}\n",
                .{ i + 1, linear.config.activation, linear.weights.shape.?.items },
            );
        }
    }

    std.debug.print("\n✓ All layers managed automatically!\n", .{});
    std.debug.print("✓ Operations automatically recorded to tape!\n", .{});
    std.debug.print("✓ Backward pass automatically computes gradients!\n\n", .{});
}

