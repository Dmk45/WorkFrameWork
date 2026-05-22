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
    /// Cached activations for readout-layer gradient accumulation (common when RNN BPTT is partial).
    lstm_encoding: ?trix.DataObject = null,
    hidden1: ?trix.DataObject = null,

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

    fn clearCache(self: *EnergyLoadForecaster) void {
        if (self.lstm_encoding) |*t| {
            t.deinit();
            self.lstm_encoding = null;
        }
        if (self.hidden1) |*t| {
            t.deinit();
            self.hidden1 = null;
        }
    }

    fn lstmLayer(self: *EnergyLoadForecaster) *layers.LSTMLayer {
        const layer = self.nn.get_layer(0) orelse unreachable;
        return &layer.lstm;
    }

    fn linearLayer(self: *EnergyLoadForecaster, idx: usize) *layers.LinearLayer {
        const layer = self.nn.get_layer(idx) orelse unreachable;
        return &layer.linear;
    }

    /// Forward: sequence of [batch, features] → [batch, 1] predicted load.
    pub fn forward(
        self: *EnergyLoadForecaster,
        allocator: std.mem.Allocator,
        sequence: []const *trix.DataObject,
    ) !trix.DataObject {
        self.clearCache();

        const lstm = self.lstmLayer();
        var lstm_out = try lstm.forwardSequence(allocator, sequence);
        lstm_out.enableGrad();
        try lstm_out.ensureGradValue();
        self.lstm_encoding = try cloneTensor(allocator, &lstm_out);

        const linear1 = self.linearLayer(1);
        var h1 = try linear1.forward(allocator, &lstm_out);
        h1.enableGrad();
        try h1.ensureGradValue();
        self.hidden1 = try cloneTensor(allocator, &h1);
        lstm_out.deinit();

        const linear2 = self.linearLayer(2);
        var output = try linear2.forward(allocator, &h1);
        output.enableGrad();
        try output.ensureGradValue();
        h1.deinit();

        return output;
    }

    fn accumulateBiasGrad(d_output: *trix.DataObject, bias: *trix.DataObject) !void {
        try bias.ensureGradValue();
        const batch = d_output.shape.?.items[0];
        const out_dim = d_output.shape.?.items[1];
        for (0..out_dim) |j| {
            var acc: f32 = 0.0;
            for (0..batch) |b| {
                acc += d_output.grad_value.?.items[b * out_dim + j];
            }
            bias.grad_value.?.items[j] += acc;
        }
    }

    /// Propagate MSE gradients into linear readout weights/bias using cached activations.
    fn accumulateReadoutGrads(self: *EnergyLoadForecaster, allocator: std.mem.Allocator, d_output: *trix.DataObject) !void {
        const h1_cache = &self.hidden1.?.*;
        const lstm_cache = &self.lstm_encoding.?.*;

        var d_loss = try trix.DataObject.init(allocator, d_output.shape.?.items, .f32);
        defer d_loss.deinit();
        @memcpy(d_loss.values.items, d_output.grad_value.?.items);

        const linear2 = self.linearLayer(2);
        try grad_math.matmulBackward(h1_cache, &linear2.weights, &d_loss, allocator);
        try accumulateBiasGrad(&d_loss, &linear2.bias);

        var w2_t = try modelwork2.core_math.transpose(allocator, &linear2.weights);
        defer w2_t.deinit();
        var d_h1 = try modelwork2.core_math.matmul(allocator, &d_loss, &w2_t);
        defer d_h1.deinit();

        const linear1 = self.linearLayer(1);
        var d_lstm = d_h1;
        if (std.mem.eql(u8, linear1.config.activation, "relu")) {
            h1_cache.enableGrad();
            try h1_cache.ensureGradValue();
            @memset(h1_cache.grad_value.?.items, 0.0);
            try grad_math.reluBackward(h1_cache, &d_h1);
            d_lstm = try trix.DataObject.init(allocator, h1_cache.shape.?.items, .f32);
            defer d_lstm.deinit();
            @memcpy(d_lstm.values.items, h1_cache.grad_value.?.items);
        }

        try grad_math.matmulBackward(lstm_cache, &linear1.weights, &d_lstm, allocator);
        try accumulateBiasGrad(&d_lstm, &linear1.bias);
    }

    /// One training step: forward → MSE → backward through head → Adam update.
    pub fn trainStep(
        self: *EnergyLoadForecaster,
        allocator: std.mem.Allocator,
        sequence: []const *trix.DataObject,
        target: *trix.DataObject,
        optimizer: *grad.Adam,
    ) !f32 {
        self.nn.zero_grad();

        var prediction = try self.forward(allocator, sequence);
        defer prediction.deinit();

        const loss = try grad.meanSquaredError(&prediction, target);

        var loss_grad = try cloneTensor(allocator, &prediction);
        defer loss_grad.deinit();
        @memcpy(loss_grad.grad_value.?.items, prediction.grad_value.?.items);

        try self.nn.backward(allocator, &loss_grad);
        try self.accumulateReadoutGrads(allocator, &prediction);

        if (self.cfg.grad_clip_norm > 0) {
            for (self.nn.layers.items) |*layer| {
                switch (layer.*) {
                    .linear => |*l| {
                        grad.clipGradientsByNorm(&l.weights, self.cfg.grad_clip_norm);
                        grad.clipGradientsByNorm(&l.bias, self.cfg.grad_clip_norm);
                    },
                    else => {},
                }
            }
        }

        try self.nn.update_parameters(optimizer);
        return loss;
    }

    pub fn deinit(self: *EnergyLoadForecaster) void {
        self.clearCache();
        self.nn.deinit();
    }
};

fn cloneTensor(allocator: std.mem.Allocator, src: *const trix.DataObject) !trix.DataObject {
    var copy = try trix.DataObject.init(allocator, src.shape.?.items, .f32);
    @memcpy(copy.values.items, src.values.items);
    if (src.grad) {
        copy.enableGrad();
        try copy.ensureGradValue();
        if (src.grad_value) |g| {
            @memcpy(copy.grad_value.?.items, g.items);
        }
    }
    return copy;
}

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

    const readout = model.linearLayer(2);
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
