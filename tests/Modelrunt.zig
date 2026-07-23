const std = @import("std");
const modelwork2 = @import("modelwork2");
const crypto_dataset = @import("crypto_dataset");

const trix = modelwork2.matrix;
const layers = modelwork2.layers;
const grad = modelwork2.grad;
const grad_math = modelwork2.grad_math;

/// Hyperparameters for crypto price prediction model
pub const CryptoConfig = struct {
    batch_size: usize = 32,
    hidden1: usize = 864,
    hidden2: usize = 432,
    hidden3: usize = 216,
    hidden4: usize = 108,
    hidden5: usize = 54,
    hidden6: usize = 27,
    learning_rate: f32 = 0.001,
    epochs: usize = 100,
    train_split: f32 = 0.8,
};

/// Crypto price forecaster: MLP for predicting BTC prices from Kalshi features
pub const CryptoForecaster = struct {
    allocator: std.mem.Allocator,
    cfg: CryptoConfig,
    nn: layers.NeuralNetwork,

    pub fn init(allocator: std.mem.Allocator, cfg: CryptoConfig, num_features: usize) !CryptoForecaster {
        var nn = try layers.NeuralNetwork.init(allocator);

        const layer1 = try layers.LinearLayer.init(
            allocator,
            num_features,
            cfg.hidden1,
            "relu",
            null,
            null,
        );
        try nn.add(layer1);

        const layer2 = try layers.LinearLayer.init(
            allocator,
            cfg.hidden1,
            cfg.hidden2,
            "relu",
            null,
            null,
        );
        try nn.add(layer2);

        const layer3 = try layers.LinearLayer.init(
            allocator,
            cfg.hidden2,
            cfg.hidden3,
            "relu",
            null,
            null,
        );
        try nn.add(layer3);

        const layer4 = try layers.LinearLayer.init(
            allocator,
            cfg.hidden3,
            cfg.hidden4,
            "relu",
            null,
            null,
        );
        try nn.add(layer4);

        const layer5 = try layers.LinearLayer.init(
            allocator,
            cfg.hidden4,
            cfg.hidden5,
            "relu",
            null,
            null,
        );
        try nn.add(layer5);

        const layer6 = try layers.LinearLayer.init(
            allocator,
            cfg.hidden5,
            cfg.hidden6,
            "relu",
            null,
            null,
        );
        try nn.add(layer6);

        const output = try layers.LinearLayer.init(
            allocator,
            cfg.hidden6,
            1,
            "none",
            null,
            null,
        );
        try nn.add(output);

        return .{
            .allocator = allocator,
            .cfg = cfg,
            .nn = nn,
        };
    }

    pub fn forward(self: *CryptoForecaster, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        var output = try self.nn.forward(allocator, .{ .tensor = input });
        output.enableGrad();
        try output.ensureGradValue();
        return output;
    }

    pub fn trainStep(
        self: *CryptoForecaster,
        allocator: std.mem.Allocator,
        input: *trix.DataObject,
        target: *trix.DataObject,
        optimizer: *grad.Adam,
    ) !f32 {
        return self.nn.trainStep(allocator, .{ .tensor = input }, target, optimizer);
    }

    pub fn deinit(self: *CryptoForecaster) void {
        self.nn.deinit();
    }
};

/// Train and evaluate crypto price prediction model
pub fn trainAndEvaluateCryptoModel(
    allocator: std.mem.Allocator,
    kalshi_path: []const u8,
    btc_path: []const u8,
) !void {
    const cfg = CryptoConfig{};
    const feature_cfg = crypto_dataset.FeatureConfig{};

    std.debug.print("Loading and aligning crypto datasets...\n", .{});
    var dataset = try crypto_dataset.parseAndAlign(allocator, kalshi_path, btc_path, feature_cfg);
    defer dataset.deinit();

    const num_samples = dataset.x_tensor.shape.?.items[0];
    const num_features = dataset.x_tensor.shape.?.items[1];

    std.debug.print("Dataset loaded: {d} samples, {d} features\n", .{ num_samples, num_features });

    // Split into train/validation
    const train_size = @as(usize, @intFromFloat(@as(f32, @floatFromInt(num_samples)) * cfg.train_split));
    const val_size = num_samples - train_size;

    std.debug.print("Train samples: {d}, Validation samples: {d}\n", .{ train_size, val_size });

    // Calculate baseline (mean prediction) for comparison
    var train_sum: f32 = 0.0;
    for (0..train_size) |i| {
        train_sum += dataset.y_tensor.values.items[i];
    }
    const train_mean = train_sum / @as(f32, @floatFromInt(train_size));

    // Calculate baseline MSE and MAE on validation set
    var baseline_mse: f32 = 0.0;
    var baseline_mae: f32 = 0.0;
    var val_count: usize = 0;
    for (train_size..num_samples) |i| {
        const diff = train_mean - dataset.y_tensor.values.items[i];
        baseline_mse += diff * diff;
        baseline_mae += if (diff < 0) -diff else diff;
        val_count += 1;
    }
    if (val_count > 0) {
        baseline_mse /= @as(f32, @floatFromInt(val_count));
        baseline_mae /= @as(f32, @floatFromInt(val_count));
    }

    std.debug.print("\n--- Baseline Performance (Mean Prediction) ---\n", .{});
    std.debug.print("Train mean BTC price: ${d:.2}\n", .{train_mean});
    std.debug.print("Baseline val_loss (MSE): {d:.6}\n", .{baseline_mse});
    std.debug.print("Baseline val_mae: ${d:.2}\n", .{baseline_mae});
    std.debug.print("----------------------------------------------\n", .{});

    // Initialize model
    var model = try CryptoForecaster.init(allocator, cfg, num_features);
    defer model.deinit();

    var optimizer = try grad.Adam.init(allocator, cfg.learning_rate, 0.9, 0.999, 1e-8);
    defer optimizer.deinit();

    // Preallocate optimizer state for all layers to avoid per-step reallocation
    try model.nn.preallocateOptimizerState(&optimizer);

    std.debug.print("\nStarting training for {d} epochs...\n", .{cfg.epochs});

    var final_val_mae: f32 = 0.0;
    var first_val_mae: f32 = 0.0;
    var first_val_loss: f32 = 0.0;

    for (0..cfg.epochs) |epoch| {
        var train_loss: f32 = 0.0;
        var train_batches: usize = 0;

        // Pre-allocate batch tensors for training (reused across batches)
        const max_batch_size = cfg.batch_size;
        var batch_input = try trix.DataObject.init(allocator, &[_]usize{ max_batch_size, num_features }, .f32);
        defer batch_input.deinit();
        var batch_target = try trix.DataObject.init(allocator, &[_]usize{ max_batch_size, 1 }, .f32);
        defer batch_target.deinit();

        // Training loop
        var batch_start: usize = 0;
        while (batch_start < train_size) {
            const batch_end = @min(batch_start + cfg.batch_size, train_size);
            const current_batch_size = batch_end - batch_start;

            // Fill batch from dataset using memcpy for efficiency
            for (0..current_batch_size) |i| {
                const sample_idx = batch_start + i;
                const src_idx = sample_idx * num_features;
                const dst_idx = i * num_features;
                @memcpy(batch_input.values.items[dst_idx .. dst_idx + num_features], dataset.x_tensor.values.items[src_idx .. src_idx + num_features]);
                batch_target.values.items[i] = dataset.y_tensor.values.items[sample_idx];
            }

            const loss = try model.trainStep(allocator, &batch_input, &batch_target, &optimizer);
            train_loss += loss;
            train_batches += 1;
            batch_start = batch_end;
        }

        const avg_train_loss = train_loss / @as(f32, @floatFromInt(train_batches));

        // Validation
        var val_loss: f32 = 0.0;
        var val_batches: usize = 0;
        var val_mae: f32 = 0.0;

        // Pre-allocate validation batch tensors
        var val_batch_input = try trix.DataObject.init(allocator, &[_]usize{ max_batch_size, num_features }, .f32);
        defer val_batch_input.deinit();
        var val_batch_target = try trix.DataObject.init(allocator, &[_]usize{ max_batch_size, 1 }, .f32);
        defer val_batch_target.deinit();

        batch_start = train_size;
        while (batch_start < num_samples) {
            const batch_end = @min(batch_start + cfg.batch_size, num_samples);
            const current_batch_size = batch_end - batch_start;

            // Fill validation batch using memcpy
            for (0..current_batch_size) |i| {
                const sample_idx = batch_start + i;
                const src_idx = sample_idx * num_features;
                const dst_idx = i * num_features;
                @memcpy(val_batch_input.values.items[dst_idx .. dst_idx + num_features], dataset.x_tensor.values.items[src_idx .. src_idx + num_features]);
                val_batch_target.values.items[i] = dataset.y_tensor.values.items[sample_idx];
            }

            var output = try model.forward(allocator, &val_batch_input);
            defer output.deinit();

            // Compute MSE loss
            var mse: f32 = 0.0;
            var mae: f32 = 0.0;
            for (0..current_batch_size) |i| {
                const diff = output.values.items[i] - val_batch_target.values.items[i];
                mse += diff * diff;
                mae += if (diff < 0) -diff else diff;
            }
            mse /= @as(f32, @floatFromInt(current_batch_size));
            mae /= @as(f32, @floatFromInt(current_batch_size));

            val_loss += mse;
            val_mae += mae;
            val_batches += 1;
            batch_start = batch_end;
        }

        const avg_val_loss = val_loss / @as(f32, @floatFromInt(val_batches));
        const avg_val_mae = val_mae / @as(f32, @floatFromInt(val_batches));

        // Store first epoch metrics
        if (epoch == 0) {
            first_val_mae = avg_val_mae;
            first_val_loss = avg_val_loss;
        }

        final_val_mae = avg_val_mae;

        std.debug.print("Epoch {d}/{d} - train_loss: {d:.6}, val_loss: {d:.6}, val_mae: ${d:.2}\n", .{
            epoch + 1,
            cfg.epochs,
            avg_train_loss,
            avg_val_loss,
            avg_val_mae,
        });

        // Show sample predictions on last epoch
        if (epoch == cfg.epochs - 1) {
            std.debug.print("\n--- Sample Predictions (Validation Set) ---\n", .{});
            const sample_count = @min(@as(usize, 5), val_size);
            var sample_idx: usize = train_size;
            while (sample_idx < train_size + sample_count) : (sample_idx += 1) {
                var sample_input = try trix.DataObject.init(allocator, &[_]usize{ 1, num_features }, .f32);
                defer sample_input.deinit();

                for (0..num_features) |f| {
                    const src_idx = sample_idx * num_features + f;
                    sample_input.values.items[f] = dataset.x_tensor.values.items[src_idx];
                }

                var sample_target = try trix.DataObject.init(allocator, &[_]usize{ 1, 1 }, .f32);
                defer sample_target.deinit();
                sample_target.values.items[0] = dataset.y_tensor.values.items[sample_idx];

                var sample_output = try model.forward(allocator, &sample_input);
                defer sample_output.deinit();

                const prediction = sample_output.values.items[0];
                const actual = sample_target.values.items[0];
                const diff = prediction - actual;

                std.debug.print("Sample {d}: Predicted: ${d:.2}, Actual: ${d:.2}, Error: ${d:.2}\n", .{
                    sample_idx - train_size + 1,
                    prediction,
                    actual,
                    diff,
                });
            }
            std.debug.print("-------------------------------------------\n", .{});
        }
    }

    std.debug.print("\n--- Final Performance Comparison ---\n", .{});
    std.debug.print("Baseline val_mae: ${d:.2}\n", .{baseline_mae});
    std.debug.print("First epoch val_mae: ${d:.2}\n", .{first_val_mae});
    std.debug.print("Final model val_mae: ${d:.2}\n", .{final_val_mae});

    const baseline_improvement = baseline_mae - final_val_mae;
    const baseline_improvement_pct = (baseline_improvement / baseline_mae) * 100.0;
    std.debug.print("Improvement over baseline: ${d:.2} ({d:.1}%)\n", .{ baseline_improvement, baseline_improvement_pct });

    const training_improvement = first_val_mae - final_val_mae;
    const training_improvement_pct = (training_improvement / first_val_mae) * 100.0;
    std.debug.print("Improvement from first epoch: ${d:.2} ({d:.1}%)\n", .{ training_improvement, training_improvement_pct });
    std.debug.print("--------------------------------------\n", .{});
    std.debug.print("Training complete!\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const kalshi_path = "kalshi_training_120d_dataset.json";
    const btc_path = "btc_price_120d_dataset.json";

    try trainAndEvaluateCryptoModel(allocator, kalshi_path, btc_path);
}
