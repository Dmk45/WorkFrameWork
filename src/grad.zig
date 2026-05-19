const std = @import("std");
const trix = @import("matrix.zig");

/// Computes the softmax of the input tensor along the last dimension
pub fn softmax(allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    // Assume input is 2D: batch_size x num_classes
    const batch_size = input.shape.?.items[0];
    const num_classes = input.shape.?.items[1];

    var output = try trix.DataObject.init(allocator, &[_]usize{ batch_size, num_classes }, .f32);

    for (0..batch_size) |b| {
        // Find max for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        for (0..num_classes) |c| {
            const val = input.get(&[_]usize{ b, c });
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        var sum_exp: f32 = 0.0;
        for (0..num_classes) |c| {
            const val = input.get(&[_]usize{ b, c });
            const exp_val = std.math.exp(val - max_val);
            try output.set(b * num_classes + c, exp_val);
            sum_exp += exp_val;
        }

        // Normalize
        for (0..num_classes) |c| {
            const idx = b * num_classes + c;
            const val = output.values.items[idx];
            output.values.items[idx] = val / sum_exp;
        }
    }

    return output;
}

/// Mean Squared Error loss
pub fn meanSquaredError(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) {
        return error.ShapeMismatch;
    }

    var loss: f32 = 0.0;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));

    for (0..y_pred.values.items.len) |i| {
        const diff = y_pred.values.items[i] - y_true.values.items[i];
        loss += diff * diff;
    }

    loss /= n;

    // Compute gradients if enabled
    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const diff = y_pred.values.items[i] - y_true.values.items[i];
            y_pred.grad_value.?.items[i] += (2.0 * diff) / n;
        }
    }

    return loss;
}

/// Cross Entropy loss (assumes y_pred is logits, y_true is one-hot encoded)
pub fn crossEntropy(y_pred: *trix.DataObject, y_true: *trix.DataObject, allocator: std.mem.Allocator) !f32 {
    // Apply softmax to y_pred
    var probs = try softmax(allocator, y_pred);
    defer probs.deinit();

    var loss: f32 = 0.0;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));

    for (0..y_pred.values.items.len) |i| {
        if (y_true.values.items[i] > 0.0) {
            loss -= y_true.values.items[i] * std.math.log(f32, std.math.e, probs.values.items[i] + 1e-7);
        }
    }

    loss /= n;

    // Compute gradients if enabled
    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            y_pred.grad_value.?.items[i] += (probs.values.items[i] - y_true.values.items[i]) / n;
        }
    }

    return loss;
}

/// Adam Optimizer
pub const Adam = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: ?std.array_list.Managed(f32),
    v: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, beta1: f32, beta2: f32, epsilon: f32) !Adam {
        return Adam{
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .t = 0,
            .m = null,
            .v = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Adam) void {
        if (self.m) |*m| m.deinit();
        if (self.v) |*v| v.deinit();
    }

    pub fn step(self: *Adam, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;

        self.t += 1;

        // Initialize m and v if not done
        if (self.m == null) {
            self.m = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.m.?.resize(param.values.items.len);
            @memset(self.m.?.items, 0.0);
        }
        if (self.v == null) {
            self.v = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.v.?.resize(param.values.items.len);
            @memset(self.v.?.items, 0.0);
        }

        const t_f = @as(f32, @floatFromInt(self.t));

        for (0..param.values.items.len) |i| {
            const g = param.grad_value.?.items[i];

            // Update biased first moment estimate
            self.m.?.items[i] = self.beta1 * self.m.?.items[i] + (1.0 - self.beta1) * g;

            // Update biased second raw moment estimate
            self.v.?.items[i] = self.beta2 * self.v.?.items[i] + (1.0 - self.beta2) * g * g;

            // Compute bias-corrected first moment estimate
            const m_hat = self.m.?.items[i] / (1.0 - std.math.pow(f32, self.beta1, t_f));

            // Compute bias-corrected second raw moment estimate
            const v_hat = self.v.?.items[i] / (1.0 - std.math.pow(f32, self.beta2, t_f));

            // Update parameter
            param.values.items[i] -= self.lr * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
        }

        // Zero gradients after update
        @memset(param.grad_value.?.items, 0.0);
    }
};

/// SGD optimizer with optional momentum and Nesterov acceleration.
pub const SGD = struct {
    lr: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    velocity: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, momentum: f32, nesterov: bool, weight_decay: f32) SGD {
        return .{
            .lr = lr,
            .momentum = momentum,
            .nesterov = nesterov,
            .weight_decay = weight_decay,
            .velocity = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SGD) void {
        if (self.velocity) |*v| v.deinit();
    }

    pub fn step(self: *SGD, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;
        if (self.velocity == null) {
            self.velocity = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.velocity.?.resize(param.values.items.len);
            @memset(self.velocity.?.items, 0.0);
        }

        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay != 0.0) grad += self.weight_decay * param.values.items[i];
            self.velocity.?.items[i] = self.momentum * self.velocity.?.items[i] + grad;
            const update = if (self.nesterov) grad + self.momentum * self.velocity.?.items[i] else self.velocity.?.items[i];
            param.values.items[i] -= self.lr * update;
        }
        @memset(param.grad_value.?.items, 0.0);
    }
};

/// RMSprop optimizer.
pub const RMSprop = struct {
    lr: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    square_avg: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, alpha: f32, epsilon: f32, weight_decay: f32) RMSprop {
        return .{
            .lr = lr,
            .alpha = alpha,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .square_avg = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RMSprop) void {
        if (self.square_avg) |*s| s.deinit();
    }

    pub fn step(self: *RMSprop, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;
        if (self.square_avg == null) {
            self.square_avg = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.square_avg.?.resize(param.values.items.len);
            @memset(self.square_avg.?.items, 0.0);
        }
        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay != 0.0) grad += self.weight_decay * param.values.items[i];
            self.square_avg.?.items[i] = self.alpha * self.square_avg.?.items[i] + (1.0 - self.alpha) * grad * grad;
            param.values.items[i] -= self.lr * grad / (std.math.sqrt(self.square_avg.?.items[i]) + self.epsilon);
        }
        @memset(param.grad_value.?.items, 0.0);
    }
};

/// AdaGrad optimizer.
pub const AdaGrad = struct {
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    sum_sq: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, epsilon: f32, weight_decay: f32) AdaGrad {
        return .{
            .lr = lr,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .sum_sq = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AdaGrad) void {
        if (self.sum_sq) |*s| s.deinit();
    }

    pub fn step(self: *AdaGrad, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;
        if (self.sum_sq == null) {
            self.sum_sq = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.sum_sq.?.resize(param.values.items.len);
            @memset(self.sum_sq.?.items, 0.0);
        }
        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay != 0.0) grad += self.weight_decay * param.values.items[i];
            self.sum_sq.?.items[i] += grad * grad;
            param.values.items[i] -= self.lr * grad / (std.math.sqrt(self.sum_sq.?.items[i]) + self.epsilon);
        }
        @memset(param.grad_value.?.items, 0.0);
    }
};

pub const StepLR = struct {
    step_size: usize,
    gamma: f32,
    pub fn step(self: StepLR, base_lr: f32, epoch: usize) f32 {
        const k = @divFloor(epoch, self.step_size);
        return base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(k)));
    }
};

pub const ExponentialLR = struct {
    gamma: f32,
    pub fn step(self: ExponentialLR, base_lr: f32, epoch: usize) f32 {
        return base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(epoch)));
    }
};

pub const WarmupLR = struct {
    warmup_epochs: usize,
    target_lr: f32,
    pub fn step(self: WarmupLR, epoch: usize) f32 {
        if (self.warmup_epochs == 0 or epoch >= self.warmup_epochs) return self.target_lr;
        const t = @as(f32, @floatFromInt(epoch + 1)) / @as(f32, @floatFromInt(self.warmup_epochs));
        return self.target_lr * t;
    }
};

pub const CosineAnnealingLR = struct {
    t_max: usize,
    eta_min: f32,
    pub fn step(self: CosineAnnealingLR, base_lr: f32, epoch: usize) f32 {
        const t_f = @as(f32, @floatFromInt(@mod(epoch, self.t_max)));
        const t_max_f = @as(f32, @floatFromInt(self.t_max));
        const cosine_factor = 0.5 * (1.0 + std.math.cos(std.math.pi * t_f / t_max_f));
        return self.eta_min + (base_lr - self.eta_min) * cosine_factor;
    }
};

pub const CyclicLR = struct {
    base_lr: f32,
    max_lr: f32,
    step_size_up: usize,
    step_size_down: usize,
    mode: []const u8, // "triangular", "triangular2", "exp_range"
    gamma: f32,
    scale_fn: ?fn (f32, f32, f32) f32,

    pub fn init(base_lr: f32, max_lr: f32, step_size_up: usize, step_size_down: usize, mode: []const u8) CyclicLR {
        return .{
            .base_lr = base_lr,
            .max_lr = max_lr,
            .step_size_up = step_size_up,
            .step_size_down = step_size_down,
            .mode = mode,
            .gamma = 1.0,
            .scale_fn = null,
        };
    }

    pub fn step(self: CyclicLR, epoch: usize) f32 {
        const cycle = @divFloor(epoch, self.step_size_up + self.step_size_down);
        const x = @as(f32, @floatFromInt(@mod(epoch, self.step_size_up + self.step_size_down)));
        const step_size_up_f = @as(f32, @floatFromInt(self.step_size_up));

        var base_lr = self.base_lr;
        if (std.mem.eql(u8, self.mode, "triangular2")) {
            base_lr = self.base_lr / std.math.pow(f32, 2.0, @as(f32, @floatFromInt(cycle)));
        } else if (std.mem.eql(u8, self.mode, "exp_range")) {
            base_lr = self.base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(epoch)));
        }

        if (x <= step_size_up_f) {
            // Ascending phase
            const scale = (self.max_lr - base_lr) * (x / step_size_up_f);
            return base_lr + scale;
        } else {
            // Descending phase
            const scale = (self.max_lr - base_lr) * ((x - step_size_up_f) / @as(f32, @floatFromInt(self.step_size_down)));
            return self.max_lr - scale;
        }
    }
};

pub const ReduceLROnPlateau = struct {
    mode: []const u8, // "min" or "max"
    factor: f32,
    patience: usize,
    threshold: f32,
    threshold_mode: []const u8, // "rel" or "abs"
    cooldown: usize,
    min_lr: f32,
    eps: f32,

    best: f32,
    num_bad_epochs: usize,
    last_epoch: usize,
    in_cooldown: bool,

    pub fn init(mode: []const u8, factor: f32, patience: usize, threshold: f32, threshold_mode: []const u8, cooldown: usize, min_lr: f32, eps: f32) ReduceLROnPlateau {
        return .{
            .mode = mode,
            .factor = factor,
            .patience = patience,
            .threshold = threshold,
            .threshold_mode = threshold_mode,
            .cooldown = cooldown,
            .min_lr = min_lr,
            .eps = eps,
            .best = if (std.mem.eql(u8, mode, "min")) std.math.inf(f32) else -std.math.inf(f32),
            .num_bad_epochs = 0,
            .last_epoch = 0,
            .in_cooldown = false,
        };
    }

    pub fn step(self: *ReduceLROnPlateau, base_lr: f32, metric: f32, epoch: usize) f32 {
        if (epoch == self.last_epoch) return base_lr;
        self.last_epoch = epoch;

        if (self.in_cooldown) {
            self.in_cooldown = false;
            self.num_bad_epochs = 0;
            return base_lr;
        }

        const is_better = if (std.mem.eql(u8, self.mode, "min"))
            metric < self.best - self.threshold
        else
            metric > self.best + self.threshold;

        if (is_better) {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if (self.num_bad_epochs >= self.patience) {
            const new_lr = @max(self.min_lr, base_lr * self.factor);
            self.num_bad_epochs = 0;
            self.in_cooldown = true;
            return new_lr;
        }

        return base_lr;
    }
};

pub fn binaryCrossEntropy(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
        const y = y_true.values.items[i];
        loss += -(y * std.math.log(f32, std.math.e, p) + (1.0 - y) * std.math.log(f32, std.math.e, 1.0 - p));
    }
    loss /= n;
    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
            const y = y_true.values.items[i];
            y_pred.grad_value.?.items[i] += (p - y) / (p * (1.0 - p) * n);
        }
    }
    return loss;
}

pub fn l1Loss(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        loss += @abs(y_pred.values.items[i] - y_true.values.items[i]);
    }
    return loss / n;
}

pub fn smoothL1Loss(y_pred: *trix.DataObject, y_true: *trix.DataObject, beta: f32) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    if (beta <= 0.0) return error.InvalidBeta;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        const d = @abs(y_pred.values.items[i] - y_true.values.items[i]);
        if (d < beta) {
            loss += 0.5 * d * d / beta;
        } else {
            loss += d - 0.5 * beta;
        }
    }
    return loss / n;
}

pub fn hingeLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;
    for (0..y_pred.values.items.len) |i| {
        const margin = 1.0 - y_true.values.items[i] * y_pred.values.items[i];
        if (margin > 0.0) loss += margin;
    }
    return loss / n;
}

/// Focal Loss for addressing class imbalance
pub fn focalLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject, alpha: f32, gamma: f32) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;

    for (0..y_pred.values.items.len) |i| {
        const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
        const y = y_true.values.items[i];
        const ce_loss = -y * std.math.log(f32, std.math.e, p) - (1.0 - y) * std.math.log(f32, std.math.e, 1.0 - p);
        const pt = if (y == 1.0) p else 1.0 - p;
        const focal_weight = std.math.pow(f32, 1.0 - pt, gamma);
        loss += alpha * focal_weight * ce_loss;
    }

    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
            const y = y_true.values.items[i];
            const pt = if (y == 1.0) p else 1.0 - p;
            const focal_weight = std.math.pow(f32, 1.0 - pt, gamma);
            const grad = alpha * focal_weight * (p - y) / (p * (1.0 - p) * n);
            y_pred.grad_value.?.items[i] += grad;
        }
    }

    return loss / n;
}

/// Label Smoothing for regularization
pub fn labelSmoothingLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject, smoothing: f32, num_classes: usize) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    const smooth_value = smoothing / @as(f32, @floatFromInt(num_classes));
    var loss: f32 = 0.0;

    for (0..y_pred.values.items.len) |i| {
        const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
        const y = y_true.values.items[i];
        const target = if (y == 1.0) 1.0 - smoothing else smooth_value;
        loss += -target * std.math.log(f32, std.math.e, p);
    }

    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const p = std.math.clamp(y_pred.values.items[i], 1e-7, 1.0 - 1e-7);
            const y = y_true.values.items[i];
            const target = if (y == 1.0) 1.0 - smoothing else smooth_value;
            y_pred.grad_value.?.items[i] += -(target / (p * n));
        }
    }

    return loss / n;
}

/// Quantile Loss for quantile regression
pub fn quantileLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject, quantile: f32) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;

    for (0..y_pred.values.items.len) |i| {
        const diff = y_true.values.items[i] - y_pred.values.items[i];
        if (diff >= 0.0) {
            loss += quantile * diff;
        } else {
            loss += (quantile - 1.0) * diff;
        }
    }

    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const diff = y_true.values.items[i] - y_pred.values.items[i];
            const grad = if (diff >= 0.0) -quantile else -(quantile - 1.0);
            y_pred.grad_value.?.items[i] += grad / n;
        }
    }

    return loss / n;
}

/// Log-Cosh Loss - smoothly approximates L1
pub fn logCoshLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;

    for (0..y_pred.values.items.len) |i| {
        const diff = y_pred.values.items[i] - y_true.values.items[i];
        loss += std.math.log(f32, std.math.e, std.math.cosh(diff));
    }

    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const diff = y_pred.values.items[i] - y_true.values.items[i];
            const grad = std.math.sinh(diff) / std.math.cosh(diff);
            y_pred.grad_value.?.items[i] += grad / n;
        }
    }

    return loss / n;
}

/// Triplet Loss for metric learning
pub fn tripletLoss(anchor: *trix.DataObject, positive: *trix.DataObject, negative: *trix.DataObject, margin: f32) !f32 {
    if (anchor.values.items.len != positive.values.items.len or anchor.values.items.len != negative.values.items.len) {
        return error.ShapeMismatch;
    }

    var pos_dist: f32 = 0.0;
    var neg_dist: f32 = 0.0;

    for (0..anchor.values.items.len) |i| {
        const pos_diff = anchor.values.items[i] - positive.values.items[i];
        const neg_diff = anchor.values.items[i] - negative.values.items[i];
        pos_dist += pos_diff * pos_diff;
        neg_dist += neg_diff * neg_diff;
    }

    pos_dist = std.math.sqrt(pos_dist);
    neg_dist = std.math.sqrt(neg_dist);
    const loss = @max(0.0, pos_dist - neg_dist + margin);

    // Compute gradients if enabled
    if (anchor.grad or positive.grad or negative.grad) {
        if (pos_dist - neg_dist + margin > 0.0) {
            const scale = 1.0 / pos_dist;
            const scale_neg = 1.0 / neg_dist;

            if (anchor.grad) {
                try anchor.ensureGradValue();
                for (0..anchor.values.items.len) |i| {
                    const pos_grad = scale * (anchor.values.items[i] - positive.values.items[i]);
                    const neg_grad = scale_neg * (anchor.values.items[i] - negative.values.items[i]);
                    anchor.grad_value.?.items[i] += pos_grad - neg_grad;
                }
            }

            if (positive.grad) {
                try positive.ensureGradValue();
                for (0..positive.values.items.len) |i| {
                    const grad = -scale * (anchor.values.items[i] - positive.values.items[i]);
                    positive.grad_value.?.items[i] += grad;
                }
            }

            if (negative.grad) {
                try negative.ensureGradValue();
                for (0..negative.values.items.len) |i| {
                    const grad = scale_neg * (anchor.values.items[i] - negative.values.items[i]);
                    negative.grad_value.?.items[i] += grad;
                }
            }
        }
    }

    return loss;
}

/// Contrastive Loss for similarity learning
pub fn contrastiveLoss(y_pred: *trix.DataObject, y_true: *trix.DataObject, margin: f32) !f32 {
    if (y_pred.values.items.len != y_true.values.items.len) return error.ShapeMismatch;
    const n = @as(f32, @floatFromInt(y_pred.values.items.len));
    var loss: f32 = 0.0;

    for (0..y_pred.values.items.len) |i| {
        const dist_sq = y_pred.values.items[i] * y_pred.values.items[i];
        const y = y_true.values.items[i];

        if (y == 1.0) {
            loss += dist_sq;
        } else {
            const margin_dist = @max(0.0, margin - std.math.sqrt(dist_sq));
            loss += margin_dist * margin_dist;
        }
    }

    if (y_pred.grad) {
        try y_pred.ensureGradValue();
        for (0..y_pred.values.items.len) |i| {
            const dist = y_pred.values.items[i];
            const y = y_true.values.items[i];

            if (y == 1.0) {
                y_pred.grad_value.?.items[i] += 2.0 * dist / n;
            } else {
                const margin_dist = @max(0.0, margin - @abs(dist));
                if (margin_dist > 0.0 and dist != 0.0) {
                    const sign_dist = if (dist > 0.0) 1.0 else -1.0;
                    y_pred.grad_value.?.items[i] += -2.0 * margin_dist * sign_dist / n;
                }
            }
        }
    }

    return loss / n;
}

/// InfoNCE Loss for contrastive learning
pub fn infoNCELoss(features: *trix.DataObject, temperature: f32) !f32 {
    const batch_size = features.shape.?.items[0];
    const feature_dim = features.shape.?.items[1];

    // Normalize features
    var normalized = try trix.DataObject.init(features.allocator, &[_]usize{ batch_size, feature_dim }, .f32);
    defer normalized.deinit();

    for (0..batch_size) |i| {
        var norm: f32 = 0.0;
        for (0..feature_dim) |j| {
            const val = features.get(&[_]usize{ i, j });
            norm += val * val;
        }
        norm = std.math.sqrt(norm) + 1e-8;

        for (0..feature_dim) |j| {
            const val = features.get(&[_]usize{ i, j });
            try normalized.set(i * feature_dim + j, val / norm);
        }
    }

    // Compute similarity matrix
    var loss: f32 = 0.0;
    for (0..batch_size) |i| {
        var numerator: f32 = 0.0;
        var denominator: f32 = 0.0;

        for (0..batch_size) |j| {
            var dot_product: f32 = 0.0;
            for (0..feature_dim) |k| {
                dot_product += normalized.get(&[_]usize{ i, k }) * normalized.get(&[_]usize{ j, k });
            }
            const similarity = dot_product / temperature;
            const exp_sim = std.math.exp(similarity);

            if (j == i) {
                numerator = exp_sim;
            }
            denominator += exp_sim;
        }

        loss -= std.math.log(f32, std.math.e, numerator / denominator);
    }

    return loss / @as(f32, @floatFromInt(batch_size));
}

pub fn clipGradientsByValue(param: *trix.DataObject, min_val: f32, max_val: f32) void {
    if (param.grad_value == null) return;
    for (param.grad_value.?.items) |*g| {
        g.* = std.math.clamp(g.*, min_val, max_val);
    }
}

pub fn clipGradientsByNorm(param: *trix.DataObject, max_norm: f32) void {
    if (param.grad_value == null or max_norm <= 0.0) return;
    var norm: f32 = 0.0;
    for (param.grad_value.?.items) |g| norm += g * g;
    norm = std.math.sqrt(norm);
    if (norm <= max_norm) return;
    const scale = max_norm / (norm + 1e-12);
    for (param.grad_value.?.items) |*g| g.* *= scale;
}

/// Accumulate external gradients into a parameter gradient buffer.
pub fn accumulateGradients(param: *trix.DataObject, grads: []const f32, average_by: usize) !void {
    if (grads.len != param.values.items.len) return error.ShapeMismatch;
    try param.ensureGradValue();
    const denom = if (average_by == 0) 1.0 else @as(f32, @floatFromInt(average_by));
    for (grads, 0..) |g, i| {
        param.grad_value.?.items[i] += g / denom;
    }
}

/// AdaBound optimizer - Adam with dynamic bounds
pub const AdaBound = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    final_lr: f32,
    gamma: f32,
    t: usize,
    m: ?std.array_list.Managed(f32),
    v: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, beta1: f32, beta2: f32, epsilon: f32, final_lr: f32, gamma: f32) !AdaBound {
        return AdaBound{
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .final_lr = final_lr,
            .gamma = gamma,
            .t = 0,
            .m = null,
            .v = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AdaBound) void {
        if (self.m) |*m| m.deinit();
        if (self.v) |*v| v.deinit();
    }

    pub fn step(self: *AdaBound, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;

        self.t += 1;

        if (self.m == null) {
            self.m = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.m.?.resize(param.values.items.len);
            @memset(self.m.?.items, 0.0);
        }
        if (self.v == null) {
            self.v = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.v.?.resize(param.values.items.len);
            @memset(self.v.?.items, 0.0);
        }

        const t_f = @as(f32, @floatFromInt(self.t));
        const lower_bound = self.final_lr * (1.0 - 1.0 / (self.gamma * t_f + 1.0));
        const upper_bound = self.final_lr * (1.0 + 1.0 / (self.gamma * t_f));

        for (0..param.values.items.len) |i| {
            const g = param.grad_value.?.items[i];

            self.m.?.items[i] = self.beta1 * self.m.?.items[i] + (1.0 - self.beta1) * g;
            self.v.?.items[i] = self.beta2 * self.v.?.items[i] + (1.0 - self.beta2) * g * g;

            const m_hat = self.m.?.items[i] / (1.0 - std.math.pow(f32, self.beta1, t_f));
            const v_hat = self.v.?.items[i] / (1.0 - std.math.pow(f32, self.beta2, t_f));

            const step_size = m_hat / (std.math.sqrt(v_hat) + self.epsilon);
            const bounded_lr = std.math.clamp(self.lr, lower_bound, upper_bound);
            param.values.items[i] -= bounded_lr * step_size;
        }

        @memset(param.grad_value.?.items, 0.0);
    }
};

/// LAMB optimizer - Layer-wise Adaptive Moments optimizer for batch training
pub const LAMB = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: usize,
    m: ?std.array_list.Managed(f32),
    v: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) !LAMB {
        return LAMB{
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .t = 0,
            .m = null,
            .v = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LAMB) void {
        if (self.m) |*m| m.deinit();
        if (self.v) |*v| v.deinit();
    }

    pub fn step(self: *LAMB, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;

        self.t += 1;

        if (self.m == null) {
            self.m = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.m.?.resize(param.values.items.len);
            @memset(self.m.?.items, 0.0);
        }
        if (self.v == null) {
            self.v = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.v.?.resize(param.values.items.len);
            @memset(self.v.?.items, 0.0);
        }

        const t_f = @as(f32, @floatFromInt(self.t));

        var weight_norm: f32 = 0.0;
        var grad_norm: f32 = 0.0;
        var trust_ratio: f32 = 1.0;

        for (0..param.values.items.len) |i| {
            const g = param.grad_value.?.items[i];

            self.m.?.items[i] = self.beta1 * self.m.?.items[i] + (1.0 - self.beta1) * g;
            self.v.?.items[i] = self.beta2 * self.v.?.items[i] + (1.0 - self.beta2) * g * g;

            weight_norm += param.values.items[i] * param.values.items[i];
            grad_norm += g * g;
        }

        weight_norm = std.math.sqrt(weight_norm);
        grad_norm = std.math.sqrt(grad_norm);

        if (weight_norm > 0.0 and grad_norm > 0.0) {
            trust_ratio = weight_norm / grad_norm;
        }

        for (0..param.values.items.len) |i| {
            const m_hat = self.m.?.items[i] / (1.0 - std.math.pow(f32, self.beta1, t_f));
            const v_hat = self.v.?.items[i] / (1.0 - std.math.pow(f32, self.beta2, t_f));

            var update = m_hat / (std.math.sqrt(v_hat) + self.epsilon);
            update = update * trust_ratio;

            if (self.weight_decay > 0.0) {
                update += self.weight_decay * param.values.items[i];
            }

            param.values.items[i] -= self.lr * update;
        }

        @memset(param.grad_value.?.items, 0.0);
    }
};

/// LARS optimizer - Layer-wise Adaptive Rate Scaling
pub const LARS = struct {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    trust_coef: f32,
    epsilon: f32,
    velocity: ?std.array_list.Managed(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, lr: f32, momentum: f32, weight_decay: f32, trust_coef: f32, epsilon: f32) LARS {
        return .{
            .lr = lr,
            .momentum = momentum,
            .weight_decay = weight_decay,
            .trust_coef = trust_coef,
            .epsilon = epsilon,
            .velocity = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LARS) void {
        if (self.velocity) |*v| v.deinit();
    }

    pub fn step(self: *LARS, param: *trix.DataObject) !void {
        if (param.grad_value == null) return;

        if (self.velocity == null) {
            self.velocity = try std.array_list.Managed(f32).initCapacity(self.allocator, param.values.items.len);
            try self.velocity.?.resize(param.values.items.len);
            @memset(self.velocity.?.items, 0.0);
        }

        var weight_norm: f32 = 0.0;
        var grad_norm: f32 = 0.0;

        for (0..param.values.items.len) |i| {
            weight_norm += param.values.items[i] * param.values.items[i];
            grad_norm += param.grad_value.?.items[i] * param.grad_value.?.items[i];
        }

        weight_norm = std.math.sqrt(weight_norm);
        grad_norm = std.math.sqrt(grad_norm);

        var local_lr = self.lr;
        if (weight_norm > 0.0 and grad_norm > 0.0) {
            local_lr = self.trust_coef * weight_norm / (grad_norm + self.weight_decay * weight_norm + self.epsilon);
        }

        for (0..param.values.items.len) |i| {
            var grad = param.grad_value.?.items[i];
            if (self.weight_decay > 0.0) {
                grad += self.weight_decay * param.values.items[i];
            }

            self.velocity.?.items[i] = self.momentum * self.velocity.?.items[i] + local_lr * grad;
            param.values.items[i] -= self.velocity.?.items[i];
        }

        @memset(param.grad_value.?.items, 0.0);
    }
};
