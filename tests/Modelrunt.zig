const std = @import("std");
const modelwork2 = @import("modelwork2");

const trix = modelwork2.matrix;
const layers = modelwork2.layers;
const grad = modelwork2.grad;
const grad_math = modelwork2.grad_math;

/// Hyperparameters for a small building energy-load forecaster.
/// Real deployment would load these from config/JSON (see `model_builder.ModelConfig`).
pub const ForecastConfig = struct {
    /// Hours of history per sample (e.g. 24h window).
    seq_len: usize = 24,
    batch_size: usize = 4,
    /// Features per timestep: ambient temp, prior load, hour-of-day (sin).
    num_features: usize = 3,
    lstm_num_layers: usize = 2,
    lstm_hidden: usize = 32,
    head_hidden: usize = 16,
    learning_rate: f32 = 0.01,
    adam_beta1: f32 = 0.9,
    adam_beta2: f32 = 0.999,
    adam_eps: f32 = 1e-8,
    /// Mini-batch training steps on synthetic meter data.
    train_steps: usize = 8,
    grad_clip_norm: f32 = 1.0,
};

/// Sequence model: stacked LSTM encoder + MLP readout → scalar next-hour load (MW).
pub const EnergyLoadForecaster = struct {
    allocator: std.mem.Allocator,
    cfg: ForecastConfig,
    nn: layers.NeuralNetwork,

    pub fn init(allocator: std.mem.Allocator, cfg: ForecastConfig) !EnergyLoadForecaster {
        var nn = try layers.NeuralNetwork.init(allocator);

        const lstm = try layers.LSTMLayer.init(
            allocator,
            cfg.lstm_num_layers,
            cfg.num_features,
            cfg.lstm_hidden,
            null,
        );
        try nn.add(lstm);

        const readout1 = try layers.LinearLayer.init(
            allocator,
            cfg.lstm_hidden,
            cfg.head_hidden,
            "relu",
            null,
            null,
        );
        try nn.add(readout1);

        const readout2 = try layers.LinearLayer.init(
            allocator,
            cfg.head_hidden,
            1,
            "none",
            null,
            null,
        );
        try nn.add(readout2);

        return .{
            .allocator = allocator,
            .cfg = cfg,
            .nn = nn,
        };
    }

    /// Forward: sequence of [batch, features] → [batch, 1] predicted load.
    pub fn forward(
        self: *EnergyLoadForecaster,
        allocator: std.mem.Allocator,
        sequence: []const *trix.DataObject,
    ) !trix.DataObject {
        var output = try self.nn.forward(allocator, .{ .sequence = sequence });
        output.enableGrad();
        try output.ensureGradValue();
        return output;
    }

    pub fn trainStep(
        self: *EnergyLoadForecaster,
        allocator: std.mem.Allocator,
        sequence: []const *trix.DataObject,
        target: *trix.DataObject,
        optimizer: *grad.Adam,
    ) !f32 {
        return self.nn.trainStep(allocator, .{ .sequence = sequence }, target, optimizer);
    }

    pub fn deinit(self: *EnergyLoadForecaster) void {
        self.nn.deinit();
    }
};

/// Synthetic hourly meter readings: smooth load curve + temperature coupling.
fn syntheticLoadAt(hour: usize, day_offset: usize) f32 {
    const t = @as(f32, @floatFromInt(hour));
    const daily = 0.6 + 0.35 * std.math.sin(t * std.math.pi / 12.0);
    const weekly = 0.05 * @as(f32, @floatFromInt(day_offset % 7));
    return daily + weekly;
}

fn buildTrainingSequence(
    allocator: std.mem.Allocator,
    cfg: ForecastConfig,
    day_offset: usize,
) !struct {
    steps: std.array_list.Managed(trix.DataObject),
    ptrs: std.array_list.Managed(*trix.DataObject),
    target: trix.DataObject,
} {
    var steps = try std.array_list.Managed(trix.DataObject).initCapacity(allocator, cfg.seq_len);
    var ptrs = try std.array_list.Managed(*trix.DataObject).initCapacity(allocator, cfg.seq_len);

    for (0..cfg.seq_len) |h| {
        const temp = 18.0 + 6.0 * std.math.sin(@as(f32, @floatFromInt(h)) * std.math.pi / 12.0);
        const load = syntheticLoadAt(h, day_offset);
        const hour_sin = std.math.sin(@as(f32, @floatFromInt(h)) * 2.0 * std.math.pi / 24.0);

        var x = try trix.DataObject.init(allocator, &[_]usize{ cfg.batch_size, cfg.num_features }, .f32);
        for (0..cfg.batch_size) |b| {
            const base = b * cfg.num_features;
            x.values.items[base + 0] = temp + @as(f32, @floatFromInt(b)) * 0.01;
            x.values.items[base + 1] = load;
            x.values.items[base + 2] = hour_sin;
        }
        try steps.append(x);
        try ptrs.append(&steps.items[steps.items.len - 1]);
    }

    const next_hour = cfg.seq_len;
    var target = try trix.DataObject.init(allocator, &[_]usize{ cfg.batch_size, 1 }, .f32);
    for (0..cfg.batch_size) |b| {
        target.values.items[b] = syntheticLoadAt(next_hour, day_offset) + @as(f32, @floatFromInt(b)) * 0.001;
    }

    return .{ .steps = steps, .ptrs = ptrs, .target = target };
}

test "energy load forecaster: train with parameter updates" {
    const allocator = std.testing.allocator;
    const cfg: ForecastConfig = .{
        .seq_len = 12,
        .batch_size = 2,
        .train_steps = 6,
        .learning_rate = 0.05,
    };

    var model = try EnergyLoadForecaster.init(allocator, cfg);
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 3), model.nn.num_layers());

    var optimizer = try grad.Adam.init(allocator, cfg.learning_rate, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps);
    defer optimizer.deinit();

    const readout_layer = model.nn.get_layer(2) orelse unreachable;
    const readout = layers.Layer.child(layers.LinearLayer, readout_layer);
    const bias_before = readout.bias.values.items[0];

    var first_loss: f32 = undefined;
    var last_loss: f32 = undefined;

    for (0..cfg.train_steps) |step| {
        var batch = try buildTrainingSequence(allocator, cfg, step);
        defer {
            for (batch.steps.items) |*s| s.deinit();
            batch.steps.deinit();
            batch.ptrs.deinit();
            batch.target.deinit();
        }

        const loss = try model.trainStep(allocator, batch.ptrs.items, &batch.target, &optimizer);
        if (step == 0) first_loss = loss;
        last_loss = loss;
        try std.testing.expect(loss >= 0.0 and std.math.isFinite(loss));
    }

    const bias_after = readout.bias.values.items[0];
    try std.testing.expect(bias_before != bias_after);
    try std.testing.expect(last_loss <= first_loss * 1.5);
}

test "energy load forecaster: forward output shape" {
    const allocator = std.testing.allocator;
    const cfg: ForecastConfig = .{ .seq_len = 5, .batch_size = 1 };

    var model = try EnergyLoadForecaster.init(allocator, cfg);
    defer model.deinit();

    var batch = try buildTrainingSequence(allocator, cfg, 0);
    defer {
        for (batch.steps.items) |*s| s.deinit();
        batch.steps.deinit();
        batch.ptrs.deinit();
        batch.target.deinit();
    }

    var output = try model.forward(allocator, batch.ptrs.items);
    defer output.deinit();

    try std.testing.expectEqualSlices(usize, &[_]usize{ cfg.batch_size, 1 }, output.shape.?.items);
}

fn customTrainStep(
    network_ptr: *anyopaque,
    allocator: std.mem.Allocator,
    input: layers.ForwardInput,
    target: *trix.DataObject,
    optimizer: *grad.Adam,
) anyerror!f32 {
    _ = network_ptr;
    _ = allocator;
    _ = input;
    _ = target;
    _ = optimizer;
    return 1.5;
}

test "neural network hooks can be overridden and cloneTensor is reusable" {
    const allocator = std.testing.allocator;

    var nn = try layers.NeuralNetwork.init(allocator);
    defer nn.deinit();

    const layer = try layers.LinearLayer.init(allocator, 1, 1, "none", null, null);
    try nn.add(layer);

    var input = try trix.DataObject.init(allocator, &[_]usize{ 1, 1 }, .f32);
    defer input.deinit();
    input.values.items[0] = 2.0;

    var clone = try trix.cloneTensor(allocator, &input);
    defer clone.deinit();
    try std.testing.expectEqual(@as(f32, 2.0), clone.values.items[0]);

    nn.setTrainStepFn(customTrainStep);

    var target = try trix.DataObject.init(allocator, &[_]usize{ 1, 1 }, .f32);
    defer target.deinit();
    target.values.items[0] = 1.0;

    var optimizer = try grad.Adam.init(allocator, 0.01, 0.9, 0.999, 1e-8);
    defer optimizer.deinit();

    const loss = try nn.trainStep(allocator, .{ .tensor = &input }, &target, &optimizer);
    try std.testing.expectEqual(@as(f32, 1.5), loss);
}
