const std = @import("std");
const modelwork2 = @import("modelwork2");
const trix = modelwork2.matrix;
const layers = modelwork2.layers;
const grad_math = modelwork2.grad_math;
const core = modelwork2.core_math;

/// Neural Network with LSTM + Linear layers using the new generic API
pub const LSTMNeuralNetwork = struct {
    allocator: std.mem.Allocator,
    nn: layers.NeuralNetwork,

    pub fn init(allocator: std.mem.Allocator) !LSTMNeuralNetwork {
        var nn = try layers.NeuralNetwork.init(allocator);
        
        // LSTM: 2 layers, input size 3, hidden size 60
        const lstm = try layers.LSTMLayer.init(allocator, 2, 3, 60, null);
        try nn.add(lstm);

        // Linear layer 1: input 60, output 10
        const linear1 = try layers.LinearLayer.init(allocator, 60, 10, "relu", null, null);
        try nn.add(linear1);

        // Linear layer 2: input 10, output 1
        const linear2 = try layers.LinearLayer.init(allocator, 10, 1, "none", null, null);
        try nn.add(linear2);

        return LSTMNeuralNetwork{
            .allocator = allocator,
            .nn = nn,
        };
    }

    /// Forward pass: input sequence -> LSTM -> Linear1 -> Linear2 -> output
    pub fn forward(self: *LSTMNeuralNetwork, allocator: std.mem.Allocator, sequence: []const *trix.DataObject) !trix.DataObject {
        // Get LSTM layer (first layer)
        const lstm_layer = &self.nn.layers.items[0];
        const lstm = &lstm_layer.lstm;
        
        // LSTM forward pass - returns final hidden state [batch, 60]
        var lstm_out = try lstm.forwardSequence(allocator, sequence);
        lstm_out.enableGrad();
        try lstm_out.ensureGradValue();

        // Linear layer 1: 60 -> 10
        const linear1_layer = &self.nn.layers.items[1];
        var linear1_out = try linear1_layer.forward(allocator, &lstm_out);
        linear1_out.enableGrad();
        try linear1_out.ensureGradValue();

        lstm_out.deinit();

        // Linear layer 2: 10 -> 1
        const linear2_layer = &self.nn.layers.items[2];
        var output = try linear2_layer.forward(allocator, &linear1_out);
        output.enableGrad();
        try output.ensureGradValue();

        linear1_out.deinit();

        return output;
    }

    /// Backward pass - compute gradients given loss gradient
    pub fn backward(self: *LSTMNeuralNetwork, allocator: std.mem.Allocator, loss_grad: *trix.DataObject, input_sequence: []const *trix.DataObject) !void {
        _ = input_sequence; // Will be used for full LSTM backward in future

        // Use the NeuralNetwork's backward method
        try self.nn.backward(allocator, loss_grad);
    }

    /// Zero all gradients
    pub fn zero_grad(self: *LSTMNeuralNetwork) void {
        self.nn.zero_grad();
    }

    pub fn deinit(self: *LSTMNeuralNetwork) void {
        self.nn.deinit();
    }
};

/// Compute Mean Squared Error loss and its gradient
pub fn mseLoss(allocator: std.mem.Allocator, output: *trix.DataObject, target: *trix.DataObject) !struct { loss: f32, grad: trix.DataObject } {
    const n = output.values.items.len;
    var loss: f32 = 0.0;

    for (0..n) |i| {
        const diff = output.values.items[i] - target.values.items[i];
        loss += diff * diff;
    }
    loss /= @as(f32, @floatFromInt(n));

    // Gradient of MSE: 2 * (output - target) / n
    var grad = try trix.DataObject.init(allocator, output.shape.?.items, .f32);
    for (0..n) |i| {
        grad.values.items[i] = 2.0 * (output.values.items[i] - target.values.items[i]) / @as(f32, @floatFromInt(n));
    }
    grad.enableGrad();

    return .{ .loss = loss, .grad = grad };
}

// Test function - just for build checking, not for running
test "LSTM Neural Network Build Test" {
    const allocator = std.testing.allocator;

    // Create the neural network
    var nn = try LSTMNeuralNetwork.init(allocator);
    defer nn.deinit();

    // Create a sample input sequence: 5 timesteps, batch=1, input_size=3
    const seq_len = 5;
    const batch = 1;
    const input_size = 3;

    var sequence = try std.ArrayList(trix.DataObject).initCapacity(allocator, seq_len);
    defer {
        for (sequence.items) |*item| {
            item.deinit();
        }
        sequence.deinit();
    }

    for (0..seq_len) |_| {
        const x = try trix.DataObject.init(allocator, &[_]usize{ batch, input_size }, .f32);
        // Initialize with some values
        for (x.values.items) |*v| {
            v.* = 0.1;
        }
        try sequence.append(x);
    }

    // Convert to pointer slice for forward pass
    var seq_ptrs = try std.ArrayList(*trix.DataObject).initCapacity(allocator, seq_len);
    defer seq_ptrs.deinit();

    for (sequence.items) |*item| {
        try seq_ptrs.append(item);
    }

    // Forward pass
    var output = try nn.forward(allocator, seq_ptrs.items);
    defer output.deinit();

    // Verify output shape: [batch, 1]
    try std.testing.expectEqual(@as(usize, 2), output.shape.?.items.len);
    try std.testing.expectEqual(@as(usize, batch), output.shape.?.items[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape.?.items[1]);

    // Create target for loss computation
    var target = try trix.DataObject.init(allocator, &[_]usize{ batch, 1 }, .f32);
    defer target.deinit();
    target.values.items[0] = 0.5;

    // Compute loss and gradient
    const loss_result = try mseLoss(allocator, &output, &target);
    defer loss_result.grad.deinit();

    // Zero gradients
    nn.zero_grad();

    // Backward pass (simplified)
    try nn.backward(allocator, &loss_result.grad, seq_ptrs.items);
}
