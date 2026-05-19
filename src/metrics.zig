const std = @import("std");
const trix = @import("matrix.zig");

const FileType = if (@hasDecl(std.fs, "File")) std.fs.File else std.Io.File;

fn initObjectMap(allocator: std.mem.Allocator) !std.json.ObjectMap {
    if (@hasDecl(std.json.ObjectMap, "empty")) {
        return .empty;
    }
    return std.json.ObjectMap.init(allocator);
}

fn deinitObjectMap(map: *std.json.ObjectMap, allocator: std.mem.Allocator) void {
    if (@hasDecl(std.json.ObjectMap, "empty")) {
        map.deinit(allocator);
    } else {
        map.deinit();
    }
}

fn putObjectMap(map: *std.json.ObjectMap, allocator: std.mem.Allocator, key: []const u8, value: std.json.Value) !void {
    if (@hasDecl(std.json.ObjectMap, "empty")) {
        try map.put(allocator, key, value);
    } else {
        try map.put(key, value);
    }
}

fn currentNanoTimestamp() i128 {
    if (@hasDecl(std.time, "nanoTimestamp")) {
        return std.time.nanoTimestamp();
    }
    return @intCast(std.Io.Clock.real.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds);
}

fn currentUnixTimestamp() i64 {
    if (@hasDecl(std.time, "timestamp")) {
        return std.time.timestamp();
    }
    return @intCast(@divFloor(currentNanoTimestamp(), std.time.ns_per_s));
}

/// Classification metrics
pub const ClassificationMetrics = struct {
    accuracy: f32,
    precision: ?f32,
    recall: ?f32,
    f1_score: ?f32,
    auc: ?f32,
    confusion_matrix: ?[][]usize,

    const PredictionPair = struct {
        pred: f32,
        label: f32,
    };

    pub fn compute(y_pred: *trix.DataObject, y_true: *trix.DataObject, threshold: f32, allocator: std.mem.Allocator) !ClassificationMetrics {
        _ = allocator;
        if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;

        const n = y_pred.values.items.len;
        var tp: usize = 0;
        var fp: usize = 0;
        var fn_count: usize = 0;
        var tn: usize = 0;

        // Basic metrics computation
        for (0..n) |i| {
            const pred_label: u32 = if (y_pred.values.items[i] >= threshold) 1 else 0;
            const true_label: u32 = if (y_true.values.items[i] >= 0.5) 1 else 0;

            if (pred_label == 1 and true_label == 1) tp += 1;
            if (pred_label == 1 and true_label == 0) fp += 1;
            if (pred_label == 0 and true_label == 1) fn_count += 1;
            if (pred_label == 0 and true_label == 0) tn += 1;
        }

        const accuracy = @as(f32, @floatFromInt(tp + tn)) / @as(f32, @floatFromInt(n));

        const precision = if (tp + fp > 0)
            @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fp))
        else
            null;

        const recall = if (tp + fn_count > 0)
            @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fn_count))
        else
            null;

        const f1_score = if (precision != null and recall != null and precision.? > 0.0 and recall.? > 0.0)
            2.0 * (precision.? * recall.?) / (precision.? + recall.?)
        else
            null;

        // Compute confusion matrix
        // No confusion matrix allocated to avoid memory leaks
        const confusion_matrix: ?[][]usize = null;
        return ClassificationMetrics{
            .accuracy = accuracy,
            .precision = precision,
            .recall = recall,
            .f1_score = f1_score,
            .auc = null, // Will be computed separately
            .confusion_matrix = confusion_matrix,
        };
    }

    /// Compute AUC using prediction scores
    pub fn computeAUC(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
        if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
        const n = y_pred.values.items.len;

        // Create pairs of (prediction, true_label)
        var pairs = try std.array_list.Managed(PredictionPair).initCapacity(y_pred.allocator, n);
        defer pairs.deinit();
        for (0..n) |i| {
            try pairs.append(.{
                .pred = y_pred.values.items[i],
                .label = y_true.values.items[i],
            });
        }

        // Sort by prediction score (ascending)
        std.mem.sort(PredictionPair, pairs.items, {}, struct {
            fn lessThan(_: void, a: PredictionPair, b: PredictionPair) bool {
                return a.pred < b.pred;
            }
        }.lessThan);

        // Compute AUC using trapezoidal rule
        var auc: f32 = 0.0;
        var prev_fpr: f32 = 0.0;
        var prev_tpr: f32 = 0.0;

        var total_pos: usize = 0;
        var total_neg: usize = 0;
        for (pairs.items) |pair| {
            if (pair.label >= 0.5) total_pos += 1 else total_neg += 1;
        }

        var pos_count: usize = 0;
        var neg_count: usize = 0;

        for (pairs.items) |pair| {
            if (pair.label >= 0.5) pos_count += 1 else neg_count += 1;

            const fpr = if (total_neg > 0) @as(f32, @floatFromInt(neg_count)) / @as(f32, @floatFromInt(total_neg)) else 0.0;
            const tpr = if (total_pos > 0) @as(f32, @floatFromInt(pos_count)) / @as(f32, @floatFromInt(total_pos)) else 0.0;

            auc += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr);

            prev_fpr = fpr;
            prev_tpr = tpr;
        }

        return auc;
    }

};




/// Progress bar for training
pub const ProgressBar = struct {
    total: usize,
    current: usize,
    width: usize,
    show_eta: bool,
    start_time: i128,

    pub fn init(total: usize, width: usize, show_eta: bool) ProgressBar {
        return .{
            .total = total,
            .current = 0,
            .width = width,
            .show_eta = show_eta,
            .start_time = currentNanoTimestamp(),
        };
    }

    pub fn update(self: *ProgressBar, current: usize) void {
        self.current = current;
        self.display();
    }

    pub fn display(self: *ProgressBar) void {
        const progress = if (self.total > 0) @as(f32, @floatFromInt(self.current)) / @as(f32, @floatFromInt(self.total)) else 0.0;
        const filled = @as(usize, @intFromFloat(progress * @as(f32, @floatFromInt(self.width))));

        var bar: [100]u8 = undefined;
        var i: usize = 0;

        while (i < filled) : (i += 1) {
            bar[i] = '=';
        }
        while (i < self.width) : (i += 1) {
            bar[i] = ' ';
        }
        bar[self.width] = 0;

        const percent = progress * 100.0;
        std.debug.print("\r[{s}]{d:.1}% ({}/{})", .{ bar[0..self.width], percent, self.current, self.total });

        if (self.show_eta and self.current > 0) {
            const current_time: i128 = currentNanoTimestamp();
            const elapsed_ns = current_time - self.start_time;
            const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
            const rate = @as(f64, @floatFromInt(self.current)) / elapsed_s;
            const remaining = @as(f64, @floatFromInt(self.total - self.current)) / rate;
            std.debug.print(" ETA: {d:.0}s", .{remaining});
        }

        if (self.current == self.total) {
            std.debug.print("\n", .{});
        }
    }
};

/// Experiment tracking
pub const ExperimentTracker = struct {
    name: []const u8,
    config: std.json.ObjectMap,
    metrics: std.json.ObjectMap,
    start_time: i128,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !ExperimentTracker {
        return .{
            .name = name,
            .config = try initObjectMap(allocator),
            .metrics = try initObjectMap(allocator),
            .start_time = currentNanoTimestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExperimentTracker) void {
        deinitObjectMap(&self.config, self.allocator);
        deinitObjectMap(&self.metrics, self.allocator);
    }

    pub fn logConfig(self: *ExperimentTracker, key: []const u8, value: std.json.Value) !void {
        try putObjectMap(&self.config, self.allocator, key, value);
    }

    pub fn logMetric(self: *ExperimentTracker, key: []const u8, value: f32) !void {
        const json_val = std.json.Value{ .float = value };
        try putObjectMap(&self.metrics, self.allocator, key, json_val);
    }

    pub fn logHyperparameter(self: *ExperimentTracker, key: []const u8, value: anytype) !void {
        const json_val = try std.json.stringifyAlloc(self.allocator, value, .{});
        defer self.allocator.free(json_val);
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_val, .{});
        defer parsed.deinit();
        try putObjectMap(&self.config, self.allocator, key, parsed.value);
    }

    pub fn save(self: *ExperimentTracker, path: []const u8) !void {
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

        const end_time: i128 = currentNanoTimestamp();
        const duration_ns = end_time - self.start_time;
        const duration_s = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;

        var experiment_obj = try initObjectMap(self.allocator);
        defer deinitObjectMap(&experiment_obj, self.allocator);

        try putObjectMap(&experiment_obj, self.allocator, "name", std.json.Value{ .string = self.name });
        try putObjectMap(&experiment_obj, self.allocator, "duration_seconds", std.json.Value{ .float = @floatCast(duration_s) });
        try putObjectMap(&experiment_obj, self.allocator, "config", std.json.Value{ .object = self.config });
        try putObjectMap(&experiment_obj, self.allocator, "metrics", std.json.Value{ .object = self.metrics });

        const json_value = std.json.Value{ .object = experiment_obj };
        const json_str = try std.json.Stringify.valueAlloc(self.allocator, json_value, .{ .whitespace = .indent_2 });
        defer self.allocator.free(json_str);

        if (@hasDecl(std.fs, "File")) {
            try file.writeAll(json_str);
        } else {
            try file.writeStreamingAll(std.Io.Threaded.global_single_threaded.io(), json_str);
        }
    }
};

/// Threshold tuning for binary classification
pub const ThresholdTuner = struct {
    pub fn findOptimalThreshold(y_pred: *trix.DataObject, y_true: *trix.DataObject, metric_type: []const u8) !f32 {
        if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;

        var best_threshold: f32 = 0.5;
        var best_score: f32 = if (std.mem.eql(u8, metric_type, "accuracy")) 0.0 else -std.math.inf(f32);

        // Test thresholds from 0.0 to 1.0 with 0.01 increments
        var threshold: f32 = 0.0;
        while (threshold <= 1.0) : (threshold += 0.01) {
            const score = try computeMetricAtThreshold(y_pred, y_true, threshold, metric_type);

            if (std.mem.eql(u8, metric_type, "accuracy")) {
                if (score > best_score) {
                    best_score = score;
                    best_threshold = threshold;
                }
            } else {
                if (score > best_score) {
                    best_score = score;
                    best_threshold = threshold;
                }
            }
        }

        return best_threshold;
    }

    fn computeMetricAtThreshold(y_pred: *trix.DataObject, y_true: *trix.DataObject, threshold: f32, metric_type: []const u8) !f32 {
        const n = y_pred.values.items.len;
        var tp: usize = 0;
        var fp: usize = 0;
        var fn_count: usize = 0;
        var tn: usize = 0;

        for (0..n) |i| {
            const pred_label = if (y_pred.values.items[i] >= threshold) 1 else 0;
            const true_label = if (y_true.values.items[i] >= 0.5) 1 else 0;

            if (pred_label == 1 and true_label == 1) tp += 1;
            if (pred_label == 1 and true_label == 0) fp += 1;
            if (pred_label == 0 and true_label == 1) fn_count += 1;
            if (pred_label == 0 and true_label == 0) tn += 1;
        }

        if (std.mem.eql(u8, metric_type, "accuracy")) {
            return @as(f32, @floatFromInt(tp + tn)) / @as(f32, @floatFromInt(n));
        } else if (std.mem.eql(u8, metric_type, "precision")) {
            return if (tp + fp > 0) @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fp)) else 0.0;
        } else if (std.mem.eql(u8, metric_type, "recall")) {
            return if (tp + fn_count > 0) @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fn_count)) else 0.0;
        } else if (std.mem.eql(u8, metric_type, "f1")) {
            const precision = if (tp + fp > 0) @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fp)) else 0.0;
            const recall = if (tp + fn_count > 0) @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fn_count)) else 0.0;
            return if (precision + recall > 0.0) 2.0 * precision * recall / (precision + recall) else 0.0;
        } else {
            return error.UnsupportedMetric;
        }
    }
};

/// Simple logging backend
pub const FileLogger = struct {
    file: FileType,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !FileLogger {
        const file = if (@hasDecl(std.fs, "File"))
            try std.fs.cwd().createFile(path, .{ .truncate = true })
        else blk: {
            const io = std.Io.Threaded.global_single_threaded.io();
            break :blk try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        };
        return .{
            .file = file,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FileLogger) void {
        if (@hasDecl(std.fs, "File")) {
            self.file.close();
        } else {
            self.file.close(std.Io.Threaded.global_single_threaded.io());
        }
    }

    pub fn log(self: *FileLogger, comptime format: []const u8, args: anytype) !void {
        const timestamp = currentUnixTimestamp();
        var buffer: [4096]u8 = undefined;
        const timestamped_msg = try std.fmt.bufPrint(&buffer, "[{d}] " ++ format, .{timestamp} ++ args);
        if (@hasDecl(std.fs, "File")) {
            try self.file.writeAll(timestamped_msg);
            try self.file.writeAll("\n");
        } else {
            const io = std.Io.Threaded.global_single_threaded.io();
            try self.file.writeStreamingAll(io, timestamped_msg);
            try self.file.writeStreamingAll(io, "\n");
        }
    }

    pub fn logMetrics(self: *FileLogger, epoch: usize, metrics: ClassificationMetrics) !void {
        try self.log("Epoch {}: accuracy={d:.3}, precision={d:.3}, recall={d:.3}, f1={d:.3}", .{ epoch, metrics.accuracy, metrics.precision orelse 0.0, metrics.recall orelse 0.0, metrics.f1_score orelse 0.0 });
    }
};
