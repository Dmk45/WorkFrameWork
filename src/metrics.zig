const std = @import("std");
const trix = @import("matrix.zig");

/// Classification metrics
pub const ClassificationMetrics = struct {
    accuracy: f32,
    precision: ?f32,
    recall: ?f32,
    f1_score: ?f32,
    auc: ?f32,
    confusion_matrix: ?[][]usize,
    
    pub fn compute(y_pred: *trix.DataObject, y_true: *trix.DataObject, threshold: f32, allocator: std.mem.Allocator) !ClassificationMetrics {
        if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
        
        const n = y_pred.values.items.len;
        var tp: usize = 0;
        var fp: usize = 0;
        var fn_count: usize = 0;
        var tn: usize = 0;
        
        // Basic metrics computation
        for (0..n) |i| {
            const pred_label = if (y_pred.values.items[i] >= threshold) 1 else 0;
            const true_label = if (y_true.values.items[i] >= 0.5) 1 else 0;
            
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
        var confusion_matrix = try allocator.alloc([]usize, 2);
        for (confusion_matrix) |*row| {
            row.* = try allocator.alloc(usize, 2);
        }
        confusion_matrix[0][0] = tn; // True Negative
        confusion_matrix[0][1] = fp; // False Positive
        confusion_matrix[1][0] = fn_count; // False Negative
        confusion_matrix[1][1] = tp; // True Positive
        
        return ClassificationMetrics{
            .accuracy = accuracy,
            .precision = precision,
            .recall = recall,
            .f1_score = f1_score,
            .auc = null, // Will be computed separately
            .confusion_matrix = confusion_matrix,
        };
    }
    
    pub fn computeAUC(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
        if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
        
        const n = y_pred.values.items.len;
        
        // Create pairs of (prediction, true_label)
        var pairs = try std.ArrayList(struct { pred: f32, label: f32 }).initCapacity(y_pred.allocator, n);
        for (0..n) |i| {
            try pairs.append(y_pred.allocator, .{
                .pred = y_pred.values.items[i],
                .label = y_true.values.items[i],
            });
        }
        
        // Sort by prediction score
        std.sort.sort(struct { pred: f32, label: f32 }, pairs.items, {}, struct {
            fn lessThan(_: void, a: struct { pred: f32, label: f32 }, b: struct { pred: f32, label: f32 }) bool {
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
    start_time: i64,
    
    pub fn init(total: usize, width: usize, show_eta: bool) ProgressBar {
        return .{
            .total = total,
            .current = 0,
            .width = width,
            .show_eta = show_eta,
            .start_time = std.time.nanoTimestamp(),
        };
    }
    
    pub fn update(self: *ProgressBar, current: usize) void {
        self.current = current;
        self.display();
    }
    
    pub fn display(self: *ProgressBar) void {
        const progress = if (self.total > 0) @as(f32, @floatFromInt(self.current)) / @as(f32, @floatFromInt(self.total)) else 0.0;
        const filled = @as(usize, @intFromFloat(progress * @as(f32, @floatFromInt(self.width))));
        const empty = self.width - filled;
        
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
        std.debug.print("\r[{}]{d:.1}% ({}/{})", .{ bar[0..self.width], percent, self.current, self.total });
        
        if (self.show_eta and self.current > 0) {
            const current_time = std.time.nanoTimestamp();
            const elapsed_ns = current_time - self.start_time;
            const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
            const rate = @as(f64, @floatFromInt(self.current)) / elapsed_s;
            const remaining = @as(f64, @floatFromInt(self.total - self.current)) / rate;
            std.debug.print(" ETA: {d:.0}s", .{remaining});
        }
        
        if (self.current == self.total) {
            std.debug.print("\n");
        }
    }
};

/// Experiment tracking
pub const ExperimentTracker = struct {
    name: []const u8,
    config: std.json.ObjectMap,
    metrics: std.json.ObjectMap,
    start_time: i64,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) !ExperimentTracker {
        return .{
            .name = name,
            .config = std.json.ObjectMap.init(allocator),
            .metrics = std.json.ObjectMap.init(allocator),
            .start_time = std.time.nanoTimestamp(),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ExperimentTracker) void {
        self.config.deinit();
        self.metrics.deinit();
    }
    
    pub fn logConfig(self: *ExperimentTracker, key: []const u8, value: std.json.Value) !void {
        try self.config.put(key, value);
    }
    
    pub fn logMetric(self: *ExperimentTracker, key: []const u8, value: f32) !void {
        const json_val = std.json.Value{ .float = value };
        try self.metrics.put(key, json_val);
    }
    
    pub fn logHyperparameter(self: *ExperimentTracker, key: []const u8, value: anytype) !void {
        const json_val = try std.json.stringifyAlloc(self.allocator, value, .{});
        defer self.allocator.free(json_val);
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_val, .{});
        defer parsed.deinit();
        try self.config.put(key, parsed.value);
    }
    
    pub fn save(self: *ExperimentTracker, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        
        const end_time = std.time.nanoTimestamp();
        const duration_ns = end_time - self.start_time;
        const duration_s = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
        
        var experiment_obj = std.json.ObjectMap.init(self.allocator);
        defer experiment_obj.deinit();
        
        try experiment_obj.put("name", std.json.Value{ .string = self.name });
        try experiment_obj.put("duration_seconds", std.json.Value{ .float = @floatCast(duration_s) });
        try experiment_obj.put("config", std.json.Value{ .object = self.config });
        try experiment_obj.put("metrics", std.json.Value{ .object = self.metrics });
        
        const json_str = try std.json.stringifyAlloc(self.allocator, experiment_obj, .{ .whitespace = .indent_2 });
        defer self.allocator.free(json_str);
        
        try file.writeAll(json_str);
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
    file: std.fs.File,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, path: []const u8) !FileLogger {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        return .{
            .file = file,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *FileLogger) void {
        self.file.close();
    }
    
    pub fn log(self: *FileLogger, comptime format: []const u8, args: anytype) !void {
        const timestamp = std.time.timestamp();
        const message = try std.fmt.allocPrint(self.allocator, "[{}] " ++ format ++ "\n", .{ timestamp } ++ args);
        defer self.allocator.free(message);
        try self.file.writeAll(message);
    }
    
    pub fn logMetrics(self: *FileLogger, epoch: usize, metrics: ClassificationMetrics) !void {
        try self.log("Epoch {}: accuracy={d:.3}, precision={d:.3}, recall={d:.3}, f1={d:.3}", 
                   .{ epoch, metrics.accuracy, metrics.precision orelse 0.0, metrics.recall orelse 0.0, metrics.f1_score orelse 0.0 });
    }
};
