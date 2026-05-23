const std = @import("std");
const trix = @import("matrix.zig");
const layers = @import("layers.zig");
const grad = @import("grad.zig");
const metrics = @import("metrics.zig");

fn currentNanoTimestamp() i128 {
    if (@hasDecl(std.time, "nanoTimestamp")) {
        return std.time.nanoTimestamp();
    }
    return @intCast(std.Io.Clock.real.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds);
}

pub const EpochStats = struct {
    train_loss: f32,
    val_loss: f32,
    metrics: ?metrics.ClassificationMetrics,
    learning_rate: f32,
    epoch_time: f64,
};

pub const TrainingHistory = struct {
    allocator: std.mem.Allocator,
    epochs: std.array_list.Managed(EpochStats),

    pub fn init(allocator: std.mem.Allocator) TrainingHistory {
        return .{ .allocator = allocator, .epochs = std.array_list.Managed(EpochStats).init(allocator) };
    }

    pub fn deinit(self: *TrainingHistory) void {
        self.epochs.deinit();
    }
};

pub const TrainerConfig = struct {
    epochs: usize = 1,
    clip_grad_norm: ?f32 = null,
    show_progress: bool = true,
    log_file: ?[]const u8 = null,
    compute_metrics: bool = true,
    threshold: f32 = 0.5,
    experiment_name: ?[]const u8 = null,
};

pub const Trainer = struct {
    allocator: std.mem.Allocator,
    model: *layers.NeuralNetwork,
    optimizer: *grad.Adam,
    config: TrainerConfig,
    history: TrainingHistory,
    best_val_loss: f32,
    logger: ?metrics.FileLogger,
    tracker: ?metrics.ExperimentTracker,
    progress_bar: ?metrics.ProgressBar,

    pub fn init(allocator: std.mem.Allocator, model: *layers.NeuralNetwork, optimizer: *grad.Adam, config: TrainerConfig) !Trainer {
        var logger: ?metrics.FileLogger = null;
        var tracker: ?metrics.ExperimentTracker = null;

        if (config.log_file) |log_path| {
            logger = try metrics.FileLogger.init(allocator, log_path);
        }

        if (config.experiment_name) |exp_name| {
            tracker = try metrics.ExperimentTracker.init(allocator, exp_name);
        }

        return .{
            .allocator = allocator,
            .model = model,
            .optimizer = optimizer,
            .config = config,
            .history = TrainingHistory.init(allocator),
            .best_val_loss = std.math.inf(f32),
            .logger = logger,
            .tracker = tracker,
            .progress_bar = null,
        };
    }

    pub fn deinit(self: *Trainer) void {
        self.history.deinit();
        if (self.logger) |*l| l.deinit();
        if (self.tracker) |*t| t.deinit();
    }

    pub fn trainEpoch(self: *Trainer, x: *trix.DataObject, y: *trix.DataObject) !f32 {
        var pred = try self.model.forward(self.allocator, .{ .tensor = x });
        defer pred.deinit();
        const loss = try grad.meanSquaredError(&pred, y);
        if (self.config.clip_grad_norm) |max_norm| {
            self.model.clip_gradients(max_norm);
        }
        try self.model.update_parameters(self.optimizer);
        return loss;
    }

    pub fn evaluate(self: *Trainer, x: *trix.DataObject, y: *trix.DataObject) !struct { loss: f32, metrics: ?metrics.ClassificationMetrics } {
        var pred = try self.model.forward(self.allocator, .{ .tensor = x });
        defer pred.deinit();
        const loss = try grad.meanSquaredError(&pred, y);

        var eval_metrics: ?metrics.ClassificationMetrics = null;
        if (self.config.compute_metrics) {
            eval_metrics = try metrics.ClassificationMetrics.compute(&pred, y, self.config.threshold, self.allocator);
            if (eval_metrics.?.auc == null) {
                const auc = try metrics.ClassificationMetrics.computeAUC(&pred, y);
                eval_metrics.?.auc = auc;
            }
        }

        return .{ .loss = loss, .metrics = eval_metrics };
    }

    pub fn fit(self: *Trainer, train_x: *trix.DataObject, train_y: *trix.DataObject, val_x: *trix.DataObject, val_y: *trix.DataObject) !void {
        if (self.config.show_progress) {
            self.progress_bar = metrics.ProgressBar.init(self.config.epochs, 50, true);
        }

        for (0..self.config.epochs) |epoch| {
            const epoch_start_time = currentNanoTimestamp();

            const train_loss = try self.trainEpoch(train_x, train_y);
            const val_result = try self.evaluate(val_x, val_y);

            if (val_result.loss < self.best_val_loss) self.best_val_loss = val_result.loss;

            const epoch_end_time = currentNanoTimestamp();
            const epoch_time = @as(f64, @floatFromInt(epoch_end_time - epoch_start_time)) / 1_000_000_000.0;

            try self.history.epochs.append(.{ .train_loss = train_loss, .val_loss = val_result.loss, .metrics = val_result.metrics, .learning_rate = self.optimizer.lr, .epoch_time = epoch_time });

            if (self.config.show_progress) {
                if (self.progress_bar) |*pb| {
                    pb.update(epoch + 1);
                }

                std.debug.print("Epoch {}/{} - train_loss: {:.4}, val_loss: {:.4}", .{ epoch + 1, self.config.epochs, train_loss, val_result.loss });
                if (val_result.metrics) |*m| {
                    std.debug.print(", accuracy: {:.4}, f1: {:.4}", .{ m.accuracy, m.f1_score orelse 0.0 });
                }
                std.debug.print("\n", .{});
            }

            if (self.logger) |*l| {
                try l.log("Epoch {}: train_loss={:.4}, val_loss={:.4}", .{ epoch + 1, train_loss, val_result.loss });
                if (val_result.metrics) |*m| {
                    try l.logMetrics(epoch + 1, m.*);
                }
            }

            if (self.tracker) |*t| {
                try t.logMetric("train_loss", train_loss);
                try t.logMetric("val_loss", val_result.loss);
                if (val_result.metrics) |*m| {
                    try t.logMetric("accuracy", m.accuracy);
                    if (m.precision) |p| try t.logMetric("precision", p);
                    if (m.recall) |r| try t.logMetric("recall", r);
                    if (m.f1_score) |f1| try t.logMetric("f1_score", f1);
                    if (m.auc) |auc| try t.logMetric("auc", auc);
                }
            }
        }

        if (self.tracker) |*t| {
            const exp_path = try std.fmt.allocPrint(self.allocator, "{s}.json", .{self.config.experiment_name.?});
            defer self.allocator.free(exp_path);
            try t.save(exp_path);
        }
    }

    pub fn saveCheckpoint(self: *Trainer, path: []const u8) !void {
        const file = if (@hasDecl(std.fs, "File"))
            try std.fs.cwd().createFile(path, .{ .truncate = true })
        else blk: {
            const io = std.Io.Threaded.global_single_threaded.io();
            break :blk try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        };
        defer if (@hasDecl(std.fs, "File"))
            file.close()
        else
            file.close(std.Io.Threaded.global_single_threaded.io());
        const payload = try std.fmt.allocPrint(self.allocator, "best_val_loss={d}\nnum_layers={}\n", .{ self.best_val_loss, self.model.layers.items.len });
        defer self.allocator.free(payload);
        if (@hasDecl(std.fs, "File")) {
            try file.writeAll(payload);
        } else {
            try file.writeStreamingAll(std.Io.Threaded.global_single_threaded.io(), payload);
        }
    }
};
