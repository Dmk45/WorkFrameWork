const std = @import("std");
const trix = @import("matrix.zig");
const core = @import("core_math.zig");
const grad_mod = @import("grad.zig");
const grad_math = @import("grad_math.zig");
const matlab_mod = @import("matlab.zig");

/// Forward function type for layers - can be customized per layer instance
pub const ForwardFn = *const fn (layer: *anyopaque, allocator: std.mem.Allocator, input: *trix.DataObject) anyerror!trix.DataObject;

/// Backprop function type for layers - handles gradient flow across layers
pub const BackpropFn = *const fn (layer: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) anyerror!trix.DataObject;

/// LSTM sequence forward function type
pub const LSTMForwardSequenceFn = *const fn (layer: *anyopaque, allocator: std.mem.Allocator, sequence: []const *trix.DataObject) anyerror!trix.DataObject;

/// Layer configuration and state management
pub const LayerConfig = struct {
    input_size: usize,
    output_size: usize,
    activation: []const u8, // "relu", "sigmoid", "tanh", "none"
};

/// Default forward implementation for LinearLayer
fn defaultLinearForward(layer_ptr: *anyopaque, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    const self: *LinearLayer = @ptrCast(@alignCast(layer_ptr));
    
    // Compute: output = input @ weights + bias
    var hidden = try core.matmul(allocator, input, &self.weights);
    hidden.enableGrad();
    try hidden.ensureGradValue();

    var output = try core.addBias(allocator, &hidden, &self.bias);
    output.enableGrad();
    try output.ensureGradValue();

    // Apply activation
    if (std.mem.eql(u8, self.config.activation, "relu")) {
        var activated = try trix.DataObject.init(allocator, output.shape.?.items, .f32);
        @memcpy(activated.values.items, output.values.items);
        matlab_mod.relu(activated.values.items.ptr, activated.values.items.len);
        activated.enableGrad();
        try activated.ensureGradValue();
        output.deinit();
        output = activated;
    } else if (std.mem.eql(u8, self.config.activation, "sigmoid")) {
        var activated = try trix.DataObject.init(allocator, output.shape.?.items, .f32);
        @memcpy(activated.values.items, output.values.items);
        matlab_mod.sigmoid(activated.values.items.ptr, activated.values.items.len);
        activated.enableGrad();
        try activated.ensureGradValue();
        output.deinit();
        output = activated;
    }

    hidden.deinit();
    return output;
}

/// Default backprop implementation for LinearLayer
fn defaultLinearBackprop(layer_ptr: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
    const self: *LinearLayer = @ptrCast(@alignCast(layer_ptr));
    
    // Backprop through activation if needed
    var grad = grad_output.*;
    if (std.mem.eql(u8, self.config.activation, "relu")) {
        try grad_math.reluBackward(&grad, &grad);
    } else if (std.mem.eql(u8, self.config.activation, "sigmoid")) {
        // Sigmoid backward would go here
    }
    
    // Compute gradient w.r.t input: grad_input = grad_output @ W^T
    var grad_input = try core.matmul(allocator, &grad, &self.weights);
    grad_input.enableGrad();
    
    // Accumulate gradients for weights: d_W = input^T @ grad_output
    // This is handled by the autodiff system
    try grad_math.matmulBackward(&self.weights, &grad, &grad, allocator);
    
    return grad_input;
}

/// Linear layer state
pub const LinearLayer = struct {
    config: LayerConfig,
    weights: trix.DataObject,
    bias: trix.DataObject,
    forward_fn: ForwardFn,
    backprop_fn: BackpropFn,

    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        activation: []const u8,
        forward_fn: ?ForwardFn,
        backprop_fn: ?BackpropFn,
    ) !LinearLayer {
        var weights = try trix.DataObject.init(allocator, &[_]usize{ input_size, output_size }, .f32);
        var bias = try trix.DataObject.init(allocator, &[_]usize{output_size}, .f32);

        // Initialize with small random values
        const nano_ts = std.time.nanoTimestamp();
        const u64_ts: u64 = @intCast(@mod(nano_ts, std.math.maxInt(i64)));
        var prng = std.Random.DefaultPrng.init(u64_ts);
        var random = prng.random();

        for (weights.values.items) |*val| {
            val.* = (random.float(f32) - 0.5) * 0.1;
        }
        for (bias.values.items) |*val| {
            val.* = 0.0;
        }

        weights.enableGrad();
        bias.enableGrad();
        try weights.ensureGradValue();
        try bias.ensureGradValue();

        return LinearLayer{
            .config = LayerConfig{
                .input_size = input_size,
                .output_size = output_size,
                .activation = activation,
            },
            .weights = weights,
            .bias = bias,
            .forward_fn = forward_fn orelse defaultLinearForward,
            .backprop_fn = backprop_fn orelse defaultLinearBackprop,
        };
    }

    /// Forward pass using the configured forward function
    pub fn forward(self: *LinearLayer, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        return self.forward_fn(self, allocator, input);
    }
    
    /// Backprop using the configured backprop function
    pub fn backprop(self: *LinearLayer, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return self.backprop_fn(self, allocator, grad_output);
    }

    pub fn deinit(self: *LinearLayer) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *LinearLayer) void {
        if (self.weights.grad_value) |*g| {
            @memset(g.items, 0.0);
        }
        if (self.bias.grad_value) |*g| {
            @memset(g.items, 0.0);
        }
    }

    pub fn get_weights(self: *LinearLayer) *trix.DataObject {
        return &self.weights;
    }

    pub fn get_bias(self: *LinearLayer) *trix.DataObject {
        return &self.bias;
    }
};

/// Layer type enum for the union
pub const LayerType = enum {
    linear,
    conv1d,
    conv2d,
    conv3d,
    lstm,
    lstm_cell,
    gru_cell,
};

/// Union type for polymorphic layer storage
pub const Layer = union(LayerType) {
    linear: LinearLayer,
    conv1d: Conv1DLayer,
    conv2d: Conv2DLayer,
    conv3d: Conv3DLayer,
    lstm: LSTMLayer,
    lstm_cell: LSTMCell,
    gru_cell: GRUCell,

    pub fn forward(self: *Layer, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        return switch (self.*) {
            .linear => |*l| l.forward(allocator, input),
            .conv1d => |*l| l.forward(allocator, input),
            .conv2d => |*l| l.forward(allocator, input),
            .conv3d => |*l| l.forward(allocator, input),
            .lstm => error.InvalidOperation, // LSTM uses forwardSequence
            .lstm_cell => |*l| {
                const batch = input.shape.?.items[0];
                var h = try trix.DataObject.init(allocator, &[_]usize{ batch, l.hidden_size }, .f32);
                @memset(h.values.items, 0.0);
                var c = try trix.DataObject.init(allocator, &[_]usize{ batch, l.hidden_size }, .f32);
                @memset(c.values.items, 0.0);
                defer h.deinit();
                defer c.deinit();
                const result = try l.forward(allocator, input, &h, &c);
                return result.h;
            },
            .gru_cell => |*l| {
                const batch = input.shape.?.items[0];
                var h = try trix.DataObject.init(allocator, &[_]usize{ batch, l.hidden_size }, .f32);
                @memset(h.values.items, 0.0);
                defer h.deinit();
                return l.forward(allocator, input, &h);
            },
        };
    }

    pub fn backprop(self: *Layer, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return switch (self.*) {
            .linear => |*l| l.backprop(allocator, grad_output),
            .conv1d => |*l| l.backprop(allocator, grad_output),
            .conv2d => |*l| l.backprop(allocator, grad_output),
            .conv3d => |*l| l.backprop(allocator, grad_output),
            .lstm => error.InvalidOperation,
            .lstm_cell => |*l| l.backprop(allocator, grad_output),
            .gru_cell => |*l| l.backprop(allocator, grad_output),
        };
    }

    pub fn deinit(self: *Layer) void {
        switch (self.*) {
            .linear => |*l| l.deinit(),
            .conv1d => |*l| l.deinit(),
            .conv2d => |*l| l.deinit(),
            .conv3d => |*l| l.deinit(),
            .lstm => |*l| l.deinit(),
            .lstm_cell => |*l| l.deinit(),
            .gru_cell => |*l| l.deinit(),
        }
    }

    pub fn zero_grad(self: *Layer) void {
        switch (self.*) {
            .linear => |*l| l.zero_grad(),
            .conv1d => |*l| l.zero_grad(),
            .conv2d => |*l| l.zero_grad(),
            .conv3d => |*l| l.zero_grad(),
            .lstm => |*l| l.zero_grad(),
            .lstm_cell => |*l| l.zero_grad(),
            .gru_cell => |*l| l.zero_grad(),
        }
    }
};

/// Simple neural network container with generic layer support
pub const NeuralNetwork = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(Layer),

    pub fn init(allocator: std.mem.Allocator) !NeuralNetwork {
        const layers = try std.ArrayList(Layer).initCapacity(allocator, 10);
        return NeuralNetwork{
            .allocator = allocator,
            .layers = layers,
        };
    }

    /// Generic add method - accepts any layer type
    pub fn add(self: *NeuralNetwork, layer: anytype) !void {
        const LayerStruct = @TypeOf(layer);
        
        // Determine which union variant to use based on type
        if (LayerStruct == LinearLayer) {
            try self.layers.append(Layer{ .linear = layer });
        } else if (LayerStruct == Conv1DLayer) {
            try self.layers.append(Layer{ .conv1d = layer });
        } else if (LayerStruct == Conv2DLayer) {
            try self.layers.append(Layer{ .conv2d = layer });
        } else if (LayerStruct == Conv3DLayer) {
            try self.layers.append(Layer{ .conv3d = layer });
        } else if (LayerStruct == LSTMLayer) {
            try self.layers.append(Layer{ .lstm = layer });
        } else if (LayerStruct == LSTMCell) {
            try self.layers.append(Layer{ .lstm_cell = layer });
        } else if (LayerStruct == GRUCell) {
            try self.layers.append(Layer{ .gru_cell = layer });
        } else {
            @compileError("Unsupported layer type: " ++ @typeName(LayerStruct));
        }
    }

    pub fn forward(self: *NeuralNetwork, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        var output = try trix.DataObject.init(allocator, input.shape.?.items, .f32);
        @memcpy(output.values.items, input.values.items);

        for (self.layers.items) |*layer| {
            const next_output = try layer.forward(allocator, &output);
            // Only deinit intermediate outputs, not the final one
            if (output.values.items.ptr != input.values.items.ptr) {
                output.deinit();
            }
            output = next_output;
        }

        return output;
    }

    /// Backward pass through all layers
    pub fn backward(self: *NeuralNetwork, allocator: std.mem.Allocator, loss_grad: *trix.DataObject) !void {
        var grad = loss_grad.*;
        
        // Backpropagate through layers in reverse order
        var i: usize = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            var layer = &self.layers.items[i];
            const prev_grad = try layer.backprop(allocator, &grad);
            if (i < self.layers.items.len - 1 or loss_grad.values.items.ptr != grad.values.items.ptr) {
                grad.deinit();
            }
            grad = prev_grad;
        }
        grad.deinit();
    }

    pub fn zero_grad(self: *NeuralNetwork) void {
        for (self.layers.items) |*layer| {
            layer.zero_grad();
        }
    }

    pub fn update_parameters(self: *NeuralNetwork, optimizer: *grad_mod.Adam) !void {
        for (self.layers.items) |*layer| {
            switch (layer.*) {
                .linear => |*l| {
                    if (l.weights.grad_value) |_| {
                        try optimizer.step(&l.weights);
                    }
                    if (l.bias.grad_value) |_| {
                        try optimizer.step(&l.bias);
                    }
                },
                else => {}, // Handle other layer types as needed
            }
        }
    }

    pub fn deinit(self: *NeuralNetwork) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit();
    }

    pub fn get_layer(self: *NeuralNetwork, idx: usize) ?*Layer {
        if (idx < self.layers.items.len) {
            return &self.layers.items[idx];
        }
        return null;
    }

    pub fn num_layers(self: *NeuralNetwork) usize {
        return self.layers.items.len;
    }
};

/// Default Conv1D forward implementation
fn defaultConv1DForward(layer_ptr: *anyopaque, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    const self: *Conv1DLayer = @ptrCast(@alignCast(layer_ptr));
    
    const s = input.shape.?.items;
    if (s.len != 3 or s[1] != self.in_channels) return error.ShapeMismatch;
    const batch = s[0];
    const width = s[2];
    const padded = width + (2 * self.padding);
    if (padded < self.kernel_size) return error.ShapeMismatch;
    const out_width = ((padded - self.kernel_size) / self.stride) + 1;
    var out = try trix.DataObject.init(allocator, &[_]usize{ batch, self.out_channels, out_width }, .f32);

    for (0..batch) |b| {
        for (0..self.out_channels) |oc| {
            for (0..out_width) |ow| {
                var acc = self.bias.values.items[oc];
                for (0..self.in_channels) |ic| {
                    for (0..self.kernel_size) |k| {
                        const x = ow * self.stride + k;
                        if (x < self.padding or x >= self.padding + width) continue;
                        const in_x = x - self.padding;
                        const in_idx = b * self.in_channels * width + ic * width + in_x;
                        const w_idx = oc * self.in_channels * self.kernel_size + ic * self.kernel_size + k;
                        acc += input.values.items[in_idx] * self.weights.values.items[w_idx];
                    }
                }
                const out_idx = b * self.out_channels * out_width + oc * out_width + ow;
                out.values.items[out_idx] = acc;
            }
        }
    }
    return out;
}

/// Default Conv1D backprop (simplified placeholder)
fn defaultConv1DBackprop(layer_ptr: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
    _ = layer_ptr;
    // Return a copy of the gradient as placeholder
    const grad_copy = try trix.DataObject.init(allocator, grad_output.shape.?.items, .f32);
    @memcpy(grad_copy.values.items, grad_output.values.items);
    return grad_copy;
}

pub const Conv1DLayer = struct {
    allocator: std.mem.Allocator,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weights: trix.DataObject, // [out_channels, in_channels, kernel_size]
    bias: trix.DataObject, // [out_channels]
    forward_fn: ForwardFn,
    backprop_fn: BackpropFn,

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, forward_fn: ?ForwardFn, backprop_fn: ?BackpropFn) !Conv1DLayer {
        const weights = try trix.DataObject.init(allocator, &[_]usize{ out_channels, in_channels, kernel_size }, .f32);
        const bias = try trix.DataObject.init(allocator, &[_]usize{out_channels}, .f32);
        for (weights.values.items) |*v| v.* = 0.01;
        for (bias.values.items) |*v| v.* = 0.0;
        return .{
            .allocator = allocator,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .weights = weights,
            .bias = bias,
            .forward_fn = forward_fn orelse defaultConv1DForward,
            .backprop_fn = backprop_fn orelse defaultConv1DBackprop,
        };
    }

    pub fn forward(self: *Conv1DLayer, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        return self.forward_fn(self, allocator, input);
    }

    pub fn backprop(self: *Conv1DLayer, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return self.backprop_fn(self, allocator, grad_output);
    }

    pub fn deinit(self: *Conv1DLayer) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *Conv1DLayer) void {
        if (self.weights.grad_value) |*g| @memset(g.items, 0.0);
        if (self.bias.grad_value) |*g| @memset(g.items, 0.0);
    }
};

/// Default Conv2D forward implementation
fn defaultConv2DForward(layer_ptr: *anyopaque, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    const self: *Conv2DLayer = @ptrCast(@alignCast(layer_ptr));
    
    const s = input.shape.?.items;
    if (s.len != 4 or s[1] != self.in_channels) return error.ShapeMismatch;
    const batch = s[0];
    const h = s[2];
    const w = s[3];
    const padded_h = h + (2 * self.padding);
    const padded_w = w + (2 * self.padding);
    if (padded_h < self.kernel_h or padded_w < self.kernel_w) return error.ShapeMismatch;
    const out_h = ((padded_h - self.kernel_h) / self.stride) + 1;
    const out_w = ((padded_w - self.kernel_w) / self.stride) + 1;
    var out = try trix.DataObject.init(allocator, &[_]usize{ batch, self.out_channels, out_h, out_w }, .f32);

    for (0..batch) |b| {
        for (0..self.out_channels) |oc| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var acc = self.bias.values.items[oc];
                    for (0..self.in_channels) |ic| {
                        for (0..self.kernel_h) |ky| {
                            for (0..self.kernel_w) |kx| {
                                const y = oy * self.stride + ky;
                                const x = ox * self.stride + kx;
                                if (y < self.padding or y >= self.padding + h) continue;
                                if (x < self.padding or x >= self.padding + w) continue;
                                const in_y = y - self.padding;
                                const in_x = x - self.padding;
                                const in_idx = b * self.in_channels * h * w + ic * h * w + in_y * w + in_x;
                                const w_idx = oc * self.in_channels * self.kernel_h * self.kernel_w +
                                    ic * self.kernel_h * self.kernel_w + ky * self.kernel_w + kx;
                                acc += input.values.items[in_idx] * self.weights.values.items[w_idx];
                            }
                        }
                    }
                    const out_idx = b * self.out_channels * out_h * out_w + oc * out_h * out_w + oy * out_w + ox;
                    out.values.items[out_idx] = acc;
                }
            }
        }
    }
    return out;
}

/// Default Conv2D backprop (simplified placeholder)
fn defaultConv2DBackprop(layer_ptr: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
    _ = layer_ptr;
    const grad_copy = try trix.DataObject.init(allocator, grad_output.shape.?.items, .f32);
    @memcpy(grad_copy.values.items, grad_output.values.items);
    return grad_copy;
}

pub const Conv2DLayer = struct {
    allocator: std.mem.Allocator,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    weights: trix.DataObject, // [out_channels, in_channels, kh, kw]
    bias: trix.DataObject, // [out_channels]
    forward_fn: ForwardFn,
    backprop_fn: BackpropFn,

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, stride: usize, padding: usize, forward_fn: ?ForwardFn, backprop_fn: ?BackpropFn) !Conv2DLayer {
        const weights = try trix.DataObject.init(allocator, &[_]usize{ out_channels, in_channels, kernel_h, kernel_w }, .f32);
        const bias = try trix.DataObject.init(allocator, &[_]usize{out_channels}, .f32);
        for (weights.values.items) |*v| v.* = 0.01;
        for (bias.values.items) |*v| v.* = 0.0;
        return .{
            .allocator = allocator,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_h = kernel_h,
            .kernel_w = kernel_w,
            .stride = stride,
            .padding = padding,
            .weights = weights,
            .bias = bias,
            .forward_fn = forward_fn orelse defaultConv2DForward,
            .backprop_fn = backprop_fn orelse defaultConv2DBackprop,
        };
    }

    pub fn forward(self: *Conv2DLayer, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        return self.forward_fn(self, allocator, input);
    }

    pub fn backprop(self: *Conv2DLayer, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return self.backprop_fn(self, allocator, grad_output);
    }

    pub fn deinit(self: *Conv2DLayer) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *Conv2DLayer) void {
        if (self.weights.grad_value) |*g| @memset(g.items, 0.0);
        if (self.bias.grad_value) |*g| @memset(g.items, 0.0);
    }
};

/// Default Conv3D forward implementation
fn defaultConv3DForward(layer_ptr: *anyopaque, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    const self: *Conv3DLayer = @ptrCast(@alignCast(layer_ptr));
    
    const s = input.shape.?.items;
    if (s.len != 5 or s[1] != self.in_channels) return error.ShapeMismatch;
    const batch = s[0];
    const d = s[2];
    const h = s[3];
    const w = s[4];
    const pd = d + (2 * self.padding);
    const ph = h + (2 * self.padding);
    const pw = w + (2 * self.padding);
    if (pd < self.kernel_d or ph < self.kernel_h or pw < self.kernel_w) return error.ShapeMismatch;
    const out_d = ((pd - self.kernel_d) / self.stride) + 1;
    const out_h = ((ph - self.kernel_h) / self.stride) + 1;
    const out_w = ((pw - self.kernel_w) / self.stride) + 1;
    var out = try trix.DataObject.init(allocator, &[_]usize{ batch, self.out_channels, out_d, out_h, out_w }, .f32);

    for (0..batch) |b| {
        for (0..self.out_channels) |oc| {
            for (0..out_d) |oz| {
                for (0..out_h) |oy| {
                    for (0..out_w) |ox| {
                        var acc = self.bias.values.items[oc];
                        for (0..self.in_channels) |ic| {
                            for (0..self.kernel_d) |kz| {
                                for (0..self.kernel_h) |ky| {
                                    for (0..self.kernel_w) |kx| {
                                        const z = oz * self.stride + kz;
                                        const y = oy * self.stride + ky;
                                        const x = ox * self.stride + kx;
                                        if (z < self.padding or z >= self.padding + d) continue;
                                        if (y < self.padding or y >= self.padding + h) continue;
                                        if (x < self.padding or x >= self.padding + w) continue;
                                        const iz = z - self.padding;
                                        const iy = y - self.padding;
                                        const ix = x - self.padding;
                                        const in_idx = b * self.in_channels * d * h * w + ic * d * h * w + iz * h * w + iy * w + ix;
                                        const w_idx = oc * self.in_channels * self.kernel_d * self.kernel_h * self.kernel_w +
                                            ic * self.kernel_d * self.kernel_h * self.kernel_w +
                                            kz * self.kernel_h * self.kernel_w + ky * self.kernel_w + kx;
                                        acc += input.values.items[in_idx] * self.weights.values.items[w_idx];
                                    }
                                }
                            }
                        }
                        const out_idx = b * self.out_channels * out_d * out_h * out_w + oc * out_d * out_h * out_w + oz * out_h * out_w + oy * out_w + ox;
                        out.values.items[out_idx] = acc;
                    }
                }
            }
        }
    }
    return out;
}

/// Default Conv3D backprop (simplified placeholder)
fn defaultConv3DBackprop(layer_ptr: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
    _ = layer_ptr;
    const grad_copy = try trix.DataObject.init(allocator, grad_output.shape.?.items, .f32);
    @memcpy(grad_copy.values.items, grad_output.values.items);
    return grad_copy;
}

pub const Conv3DLayer = struct {
    allocator: std.mem.Allocator,
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    weights: trix.DataObject, // [out_channels, in_channels, kd, kh, kw]
    bias: trix.DataObject, // [out_channels]
    forward_fn: ForwardFn,
    backprop_fn: BackpropFn,

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_d: usize, kernel_h: usize, kernel_w: usize, stride: usize, padding: usize, forward_fn: ?ForwardFn, backprop_fn: ?BackpropFn) !Conv3DLayer {
        const weights = try trix.DataObject.init(allocator, &[_]usize{ out_channels, in_channels, kernel_d, kernel_h, kernel_w }, .f32);
        const bias = try trix.DataObject.init(allocator, &[_]usize{out_channels}, .f32);
        for (weights.values.items) |*v| v.* = 0.01;
        for (bias.values.items) |*v| v.* = 0.0;
        return .{
            .allocator = allocator,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_d = kernel_d,
            .kernel_h = kernel_h,
            .kernel_w = kernel_w,
            .stride = stride,
            .padding = padding,
            .weights = weights,
            .bias = bias,
            .forward_fn = forward_fn orelse defaultConv3DForward,
            .backprop_fn = backprop_fn orelse defaultConv3DBackprop,
        };
    }

    pub fn forward(self: *Conv3DLayer, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        return self.forward_fn(self, allocator, input);
    }

    pub fn backprop(self: *Conv3DLayer, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return self.backprop_fn(self, allocator, grad_output);
    }

    pub fn deinit(self: *Conv3DLayer) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *Conv3DLayer) void {
        if (self.weights.grad_value) |*g| @memset(g.items, 0.0);
        if (self.bias.grad_value) |*g| @memset(g.items, 0.0);
    }
};

pub const PoolType = enum { max, avg };

pub fn pool2d(allocator: std.mem.Allocator, input: *trix.DataObject, kernel: usize, stride: usize, kind: PoolType) !trix.DataObject {
    const s = input.shape.?.items;
    if (s.len != 4) return error.ShapeMismatch;
    const batch = s[0];
    const channels = s[1];
    const h = s[2];
    const w = s[3];
    if (h < kernel or w < kernel) return error.ShapeMismatch;
    const out_h = ((h - kernel) / stride) + 1;
    const out_w = ((w - kernel) / stride) + 1;
    var out = try trix.DataObject.init(allocator, &[_]usize{ batch, channels, out_h, out_w }, .f32);
    for (0..batch) |b| {
        for (0..channels) |c| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var acc: f32 = 0.0;
                    var maxv: f32 = -std.math.inf(f32);
                    for (0..kernel) |ky| {
                        for (0..kernel) |kx| {
                            const iy = oy * stride + ky;
                            const ix = ox * stride + kx;
                            const idx = b * channels * h * w + c * h * w + iy * w + ix;
                            const v = input.values.items[idx];
                            acc += v;
                            if (v > maxv) maxv = v;
                        }
                    }
                    const out_idx = b * channels * out_h * out_w + c * out_h * out_w + oy * out_w + ox;
                    out.values.items[out_idx] = if (kind == .avg) acc / @as(f32, @floatFromInt(kernel * kernel)) else maxv;
                }
            }
        }
    }
    return out;
}

pub fn flatten(allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
    const s = input.shape.?.items;
    if (s.len < 2) return error.ShapeMismatch;
    const batch = s[0];
    const features = input.values.items.len / batch;
    const out = try trix.DataObject.init(allocator, &[_]usize{ batch, features }, .f32);
    @memcpy(out.values.items, input.values.items);
    return out;
}

pub fn reshapeDynamic(input: *trix.DataObject, new_shape: []const usize) !void {
    try core.reshape(input, new_shape);
}

pub fn dropout(allocator: std.mem.Allocator, input: *trix.DataObject, drop_prob: f32, training: bool, seed: u64) !trix.DataObject {
    var out = try trix.DataObject.init(allocator, input.shape.?.items, .f32);
    if (!training or drop_prob <= 0.0) {
        @memcpy(out.values.items, input.values.items);
        return out;
    }
    const keep = 1.0 - drop_prob;
    var prng = std.Random.DefaultPrng.init(seed);
    var rng = prng.random();
    for (0..input.values.items.len) |i| {
        const m: f32 = if (rng.float(f32) < drop_prob) 0.0 else 1.0 / keep;
        out.values.items[i] = input.values.items[i] * m;
    }
    return out;
}

pub fn batchNorm(allocator: std.mem.Allocator, input: *trix.DataObject, eps: f32) !trix.DataObject {
    const s = input.shape.?.items;
    if (s.len != 2) return error.ShapeMismatch;
    const batch = s[0];
    const features = s[1];
    var out = try trix.DataObject.init(allocator, s, .f32);
    for (0..features) |f| {
        var mean: f32 = 0.0;
        for (0..batch) |b| mean += input.values.items[b * features + f];
        mean /= @as(f32, @floatFromInt(batch));
        var var_: f32 = 0.0;
        for (0..batch) |b| {
            const d = input.values.items[b * features + f] - mean;
            var_ += d * d;
        }
        var_ /= @as(f32, @floatFromInt(batch));
        const inv = 1.0 / std.math.sqrt(var_ + eps);
        for (0..batch) |b| {
            const idx = b * features + f;
            out.values.items[idx] = (input.values.items[idx] - mean) * inv;
        }
    }
    return out;
}

pub fn layerNorm(allocator: std.mem.Allocator, input: *trix.DataObject, eps: f32) !trix.DataObject {
    const s = input.shape.?.items;
    if (s.len != 2) return error.ShapeMismatch;
    const batch = s[0];
    const features = s[1];
    var out = try trix.DataObject.init(allocator, s, .f32);
    for (0..batch) |b| {
        var mean: f32 = 0.0;
        for (0..features) |f| mean += input.values.items[b * features + f];
        mean /= @as(f32, @floatFromInt(features));
        var var_: f32 = 0.0;
        for (0..features) |f| {
            const d = input.values.items[b * features + f] - mean;
            var_ += d * d;
        }
        var_ /= @as(f32, @floatFromInt(features));
        const inv = 1.0 / std.math.sqrt(var_ + eps);
        for (0..features) |f| {
            const idx = b * features + f;
            out.values.items[idx] = (input.values.items[idx] - mean) * inv;
        }
    }
    return out;
}

pub fn groupNorm(allocator: std.mem.Allocator, input: *trix.DataObject, groups: usize, eps: f32) !trix.DataObject {
    const s = input.shape.?.items;
    if (s.len != 2 or groups == 0) return error.ShapeMismatch;
    const batch = s[0];
    const features = s[1];
    if (features % groups != 0) return error.ShapeMismatch;
    const group_size = features / groups;
    var out = try trix.DataObject.init(allocator, s, .f32);
    for (0..batch) |b| {
        for (0..groups) |g| {
            const start = g * group_size;
            const end = start + group_size;
            var mean: f32 = 0.0;
            for (start..end) |f| mean += input.values.items[b * features + f];
            mean /= @as(f32, @floatFromInt(group_size));
            var var_: f32 = 0.0;
            for (start..end) |f| {
                const d = input.values.items[b * features + f] - mean;
                var_ += d * d;
            }
            var_ /= @as(f32, @floatFromInt(group_size));
            const inv = 1.0 / std.math.sqrt(var_ + eps);
            for (start..end) |f| {
                const idx = b * features + f;
                out.values.items[idx] = (input.values.items[idx] - mean) * inv;
            }
        }
    }
    return out;
}

/// LSTMCell forward function type
pub const LSTMCellForwardFn = *const fn (layer: *anyopaque, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: *trix.DataObject, c_prev: *trix.DataObject) anyerror!struct { h: trix.DataObject, c: trix.DataObject };

/// Default LSTMCell forward implementation
fn defaultLSTMCellForward(layer_ptr: *anyopaque, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: *trix.DataObject, c_prev: *trix.DataObject) !struct { h: trix.DataObject, c: trix.DataObject } {
    const self: *LSTMCell = @ptrCast(@alignCast(layer_ptr));
    
    var x_proj = try core.matmul(allocator, x, &self.w_ih);
    defer x_proj.deinit();
    var h_proj = try core.matmul(allocator, h_prev, &self.w_hh);
    defer h_proj.deinit();
    var gates = try core.add(allocator, &x_proj, &h_proj);
    defer gates.deinit();
    var gates_b = try core.addBias(allocator, &gates, &self.bias);
    defer gates_b.deinit();

    const batch = x.shape.?.items[0];
    var h = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);
    var c = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);

    for (0..batch) |b| {
        for (0..self.hidden_size) |i| {
            const base = b * (4 * self.hidden_size);
            const ii = sigmoidScalar(gates_b.values.items[base + i]);
            const ff = sigmoidScalar(gates_b.values.items[base + self.hidden_size + i]);
            const gg = std.math.tanh(gates_b.values.items[base + 2 * self.hidden_size + i]);
            const oo = sigmoidScalar(gates_b.values.items[base + 3 * self.hidden_size + i]);
            const prev_c = c_prev.values.items[b * self.hidden_size + i];
            const c_val = ff * prev_c + ii * gg;
            c.values.items[b * self.hidden_size + i] = c_val;
            h.values.items[b * self.hidden_size + i] = oo * std.math.tanh(c_val);
        }
    }
    return .{ .h = h, .c = c };
}

/// Default LSTMCell backprop (placeholder)
fn defaultLSTMCellBackprop(layer_ptr: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
    _ = layer_ptr;
    const grad_copy = try trix.DataObject.init(allocator, grad_output.shape.?.items, .f32);
    @memcpy(grad_copy.values.items, grad_output.values.items);
    return grad_copy;
}

pub const LSTMCell = struct {
    input_size: usize,
    hidden_size: usize,
    w_ih: trix.DataObject, // [input_size, 4*hidden]
    w_hh: trix.DataObject, // [hidden, 4*hidden]
    bias: trix.DataObject, // [4*hidden]
    forward_fn: LSTMCellForwardFn,
    backprop_fn: BackpropFn,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, forward_fn: ?LSTMCellForwardFn, backprop_fn: ?BackpropFn) !LSTMCell {
        const w_ih = try trix.DataObject.init(allocator, &[_]usize{ input_size, 4 * hidden_size }, .f32);
        const w_hh = try trix.DataObject.init(allocator, &[_]usize{ hidden_size, 4 * hidden_size }, .f32);
        const bias = try trix.DataObject.init(allocator, &[_]usize{4 * hidden_size}, .f32);
        for (w_ih.values.items) |*v| v.* = 0.01;
        for (w_hh.values.items) |*v| v.* = 0.01;
        for (bias.values.items) |*v| v.* = 0.0;
        return .{ 
            .input_size = input_size, 
            .hidden_size = hidden_size, 
            .w_ih = w_ih, 
            .w_hh = w_hh, 
            .bias = bias,
            .forward_fn = forward_fn orelse defaultLSTMCellForward,
            .backprop_fn = backprop_fn orelse defaultLSTMCellBackprop,
        };
    }

    pub fn forward(self: *LSTMCell, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: *trix.DataObject, c_prev: *trix.DataObject) !struct { h: trix.DataObject, c: trix.DataObject } {
        return self.forward_fn(self, allocator, x, h_prev, c_prev);
    }

    pub fn backprop(self: *LSTMCell, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return self.backprop_fn(self, allocator, grad_output);
    }

    pub fn deinit(self: *LSTMCell) void {
        self.w_ih.deinit();
        self.w_hh.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *LSTMCell) void {
        if (self.w_ih.grad_value) |*g| @memset(g.items, 0.0);
        if (self.w_hh.grad_value) |*g| @memset(g.items, 0.0);
        if (self.bias.grad_value) |*g| @memset(g.items, 0.0);
    }
};

/// GRUCell forward function type
pub const GRUCellForwardFn = *const fn (layer: *anyopaque, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: *trix.DataObject) anyerror!trix.DataObject;

/// Default GRUCell forward implementation
fn defaultGRUCellForward(layer_ptr: *anyopaque, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: *trix.DataObject) !trix.DataObject {
    const self: *GRUCell = @ptrCast(@alignCast(layer_ptr));
    
    var x_proj = try core.matmul(allocator, x, &self.w_ih);
    defer x_proj.deinit();
    var h_proj = try core.matmul(allocator, h_prev, &self.w_hh);
    defer h_proj.deinit();
    var gates = try core.add(allocator, &x_proj, &h_proj);
    defer gates.deinit();
    var gates_b = try core.addBias(allocator, &gates, &self.bias);
    defer gates_b.deinit();

    const batch = x.shape.?.items[0];
    var h = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);
    for (0..batch) |b| {
        for (0..self.hidden_size) |i| {
            const base = b * (3 * self.hidden_size);
            const z = sigmoidScalar(gates_b.values.items[base + i]);
            const r = sigmoidScalar(gates_b.values.items[base + self.hidden_size + i]);
            const n_pre = gates_b.values.items[base + 2 * self.hidden_size + i] + r * h_prev.values.items[b * self.hidden_size + i];
            const n = std.math.tanh(n_pre);
            const prev = h_prev.values.items[b * self.hidden_size + i];
            h.values.items[b * self.hidden_size + i] = (1.0 - z) * n + z * prev;
        }
    }
    return h;
}

/// Default GRUCell backprop (placeholder)
fn defaultGRUCellBackprop(layer_ptr: *anyopaque, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
    _ = layer_ptr;
    const grad_copy = try trix.DataObject.init(allocator, grad_output.shape.?.items, .f32);
    @memcpy(grad_copy.values.items, grad_output.values.items);
    return grad_copy;
}

pub const GRUCell = struct {
    input_size: usize,
    hidden_size: usize,
    w_ih: trix.DataObject, // [input, 3*hidden]
    w_hh: trix.DataObject, // [hidden, 3*hidden]
    bias: trix.DataObject, // [3*hidden]
    forward_fn: GRUCellForwardFn,
    backprop_fn: BackpropFn,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, forward_fn: ?GRUCellForwardFn, backprop_fn: ?BackpropFn) !GRUCell {
        const w_ih = try trix.DataObject.init(allocator, &[_]usize{ input_size, 3 * hidden_size }, .f32);
        const w_hh = try trix.DataObject.init(allocator, &[_]usize{ hidden_size, 3 * hidden_size }, .f32);
        const bias = try trix.DataObject.init(allocator, &[_]usize{3 * hidden_size}, .f32);
        for (w_ih.values.items) |*v| v.* = 0.01;
        for (w_hh.values.items) |*v| v.* = 0.01;
        for (bias.values.items) |*v| v.* = 0.0;
        return .{ 
            .input_size = input_size, 
            .hidden_size = hidden_size, 
            .w_ih = w_ih, 
            .w_hh = w_hh, 
            .bias = bias,
            .forward_fn = forward_fn orelse defaultGRUCellForward,
            .backprop_fn = backprop_fn orelse defaultGRUCellBackprop,
        };
    }

    pub fn forward(self: *GRUCell, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: *trix.DataObject) !trix.DataObject {
        return self.forward_fn(self, allocator, x, h_prev);
    }

    pub fn backprop(self: *GRUCell, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        return self.backprop_fn(self, allocator, grad_output);
    }

    pub fn deinit(self: *GRUCell) void {
        self.w_ih.deinit();
        self.w_hh.deinit();
        self.bias.deinit();
    }

    pub fn zero_grad(self: *GRUCell) void {
        if (self.w_ih.grad_value) |*g| @memset(g.items, 0.0);
        if (self.w_hh.grad_value) |*g| @memset(g.items, 0.0);
        if (self.bias.grad_value) |*g| @memset(g.items, 0.0);
    }
};

/// Default LSTM forward sequence implementation
fn defaultLSTMForwardSequence(layer_ptr: *anyopaque, allocator: std.mem.Allocator, sequence: []const *trix.DataObject) !trix.DataObject {
    const self: *LSTMLayer = @ptrCast(@alignCast(layer_ptr));
    
    // sequence is an array of [batch, input_size] tensors
    if (sequence.len == 0) return error.EmptySequence;

    const batch = sequence[0].shape.?.items[0];

    // Initialize hidden and cell states to zeros
    var h_states = try std.ArrayList(trix.DataObject).initCapacity(allocator, self.num_layers);
    var c_states = try std.ArrayList(trix.DataObject).initCapacity(allocator, self.num_layers);
    defer h_states.deinit();
    defer c_states.deinit();

    for (0..self.num_layers) |_| {
        const h = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);
        @memset(h.values.items, 0.0);
        try h_states.append(h);

        const c = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);
        @memset(c.values.items, 0.0);
        try c_states.append(c);
    }

    // Convert to pointers for the forward pass
    var h_ptrs = try std.ArrayList(*trix.DataObject).initCapacity(allocator, self.num_layers);
    var c_ptrs = try std.ArrayList(*trix.DataObject).initCapacity(allocator, self.num_layers);
    defer h_ptrs.deinit();
    defer c_ptrs.deinit();

    for (0..self.num_layers) |i| {
        try h_ptrs.append(&h_states.items[i]);
        try c_ptrs.append(&c_states.items[i]);
    }

    // Process each timestep
    for (sequence) |x_t| {
        const h_ptr_slice = h_ptrs.items;
        const c_ptr_slice = c_ptrs.items;

        const result = try self.forward(allocator, x_t, h_ptr_slice, c_ptr_slice);

        // Update states for next timestep
        for (0..self.num_layers) |i| {
            h_states.items[i].deinit();
            c_states.items[i].deinit();
            h_states.items[i] = result.h_states.items[i];
            c_states.items[i] = result.c_states.items[i];
        }

        // Clean up intermediate states - the final ones are stored in h_states/c_states
        result.h_states.deinit();
        result.c_states.deinit();

        // Update pointers
        h_ptrs.clearRetainingCapacity();
        c_ptrs.clearRetainingCapacity();
        for (0..self.num_layers) |i| {
            try h_ptrs.append(&h_states.items[i]);
            try c_ptrs.append(&c_states.items[i]);
        }
    }

    // Return final hidden state of last layer
    const final_h = h_states.items[self.num_layers - 1];
    
    // Clean up states except the final one we're returning
    for (0..self.num_layers - 1) |i| {
        h_states.items[i].deinit();
        c_states.items[i].deinit();
    }
    c_states.items[self.num_layers - 1].deinit();
    
    return final_h;
}

/// Multi-layer LSTM that processes sequences through stacked LSTM cells
pub const LSTMLayer = struct {
    allocator: std.mem.Allocator,
    num_layers: usize,
    input_size: usize,
    hidden_size: usize,
    cells: std.ArrayList(LSTMCell),
    forward_sequence_fn: LSTMForwardSequenceFn,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, input_size: usize, hidden_size: usize, forward_sequence_fn: ?LSTMForwardSequenceFn) !LSTMLayer {
        var cells = try std.ArrayList(LSTMCell).initCapacity(allocator, num_layers);

        // First layer takes input_size, subsequent layers take hidden_size
        for (0..num_layers) |i| {
            const layer_input_size = if (i == 0) input_size else hidden_size;
            const cell = try LSTMCell.init(allocator, layer_input_size, hidden_size, null, null);
            try cells.append(cell);
        }

        return LSTMLayer{
            .allocator = allocator,
            .num_layers = num_layers,
            .input_size = input_size,
            .hidden_size = hidden_size,
            .cells = cells,
            .forward_sequence_fn = forward_sequence_fn orelse defaultLSTMForwardSequence,
        };
    }

    /// Forward pass through all LSTM layers for one timestep
    /// Returns final hidden state and cell state
    pub fn forward(self: *LSTMLayer, allocator: std.mem.Allocator, x: *trix.DataObject, h_prev: ?[]*trix.DataObject, c_prev: ?[]*trix.DataObject) !struct { h: trix.DataObject, c: trix.DataObject, h_states: std.ArrayList(trix.DataObject), c_states: std.ArrayList(trix.DataObject) } {
        const batch = x.shape.?.items[0];

        // Initialize or use provided hidden/cell states
        var h_states = try std.ArrayList(trix.DataObject).initCapacity(allocator, self.num_layers);
        var c_states = try std.ArrayList(trix.DataObject).initCapacity(allocator, self.num_layers);

        var current_input = x;
        var prev_h: *trix.DataObject = undefined;
        var prev_c: *trix.DataObject = undefined;

        // Temporary storage for intermediate results that need cleanup
        var temp_inputs = std.ArrayList(trix.DataObject).init(allocator);
        defer temp_inputs.deinit();

        for (0..self.num_layers) |layer| {
            // Initialize hidden and cell states if not provided
            var h_init: trix.DataObject = undefined;
            var c_init: trix.DataObject = undefined;

            if (h_prev) |hp| {
                h_init = hp[layer].*;
            } else {
                h_init = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);
                @memset(h_init.values.items, 0.0);
                try temp_inputs.append(h_init);
            }

            if (c_prev) |cp| {
                c_init = cp[layer].*;
            } else {
                c_init = try trix.DataObject.init(allocator, &[_]usize{ batch, self.hidden_size }, .f32);
                @memset(c_init.values.items, 0.0);
                try temp_inputs.append(c_init);
            }

            prev_h = &h_init;
            prev_c = &c_init;

            // Forward through this LSTM cell
            const result = try self.cells.items[layer].forward(allocator, current_input, prev_h, prev_c);

            try h_states.append(result.h);
            try c_states.append(result.c);

            // Output of this layer becomes input to next layer
            current_input = &h_states.items[layer];
        }

        // Return final hidden and cell states, plus all intermediate states
        const final_h = h_states.items[self.num_layers - 1];
        const final_c = c_states.items[self.num_layers - 1];

        return .{
            .h = final_h,
            .c = final_c,
            .h_states = h_states,
            .c_states = c_states,
        };
    }

    /// Process a full sequence through the LSTM
    pub fn forwardSequence(self: *LSTMLayer, allocator: std.mem.Allocator, sequence: []const *trix.DataObject) !trix.DataObject {
        return self.forward_sequence_fn(self, allocator, sequence);
    }

    pub fn deinit(self: *LSTMLayer) void {
        for (self.cells.items) |*cell| {
            cell.deinit();
        }
        self.cells.deinit();
    }

    pub fn zero_grad(self: *LSTMLayer) void {
        for (self.cells.items) |*cell| {
            cell.zero_grad();
        }
    }

    pub fn backprop(self: *LSTMLayer, allocator: std.mem.Allocator, grad_output: *trix.DataObject) !trix.DataObject {
        _ = self;
        // Simplified backprop for LSTM - would need full BPTT implementation
        const grad_copy = try trix.DataObject.init(allocator, grad_output.shape.?.items, .f32);
        @memcpy(grad_copy.values.items, grad_output.values.items);
        return grad_copy;
    }
};

pub const EmbeddingLayer = struct {
    embedding_table: trix.DataObject, // [vocab_size, embed_dim]
    vocab_size: usize,
    embed_dim: usize,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embed_dim: usize) !EmbeddingLayer {
        const t = try trix.DataObject.init(allocator, &[_]usize{ vocab_size, embed_dim }, .f32);
        for (t.values.items) |*v| v.* = 0.01;
        return .{ .embedding_table = t, .vocab_size = vocab_size, .embed_dim = embed_dim };
    }

    pub fn forward(self: *EmbeddingLayer, allocator: std.mem.Allocator, token_ids: []const usize) !trix.DataObject {
        var out = try trix.DataObject.init(allocator, &[_]usize{ token_ids.len, self.embed_dim }, .f32);
        for (token_ids, 0..) |token, i| {
            if (token >= self.vocab_size) return error.IndexOutOfBounds;
            const src = token * self.embed_dim;
            const dst = i * self.embed_dim;
            @memcpy(out.values.items[dst .. dst + self.embed_dim], self.embedding_table.values.items[src .. src + self.embed_dim]);
        }
        return out;
    }

    pub fn addPositionalEncoding(self: *EmbeddingLayer, embedding: *trix.DataObject) void {
        const len = embedding.shape.?.items[0];
        for (0..len) |pos| {
            for (0..self.embed_dim) |i| {
                const div = std.math.pow(f32, 10000.0, @as(f32, @floatFromInt((2 * (i / 2)))) / @as(f32, @floatFromInt(self.embed_dim)));
                const p = @as(f32, @floatFromInt(pos)) / div;
                const pe: f32 = if (i % 2 == 0) std.math.sin(p) else std.math.cos(p);
                embedding.values.items[pos * self.embed_dim + i] += pe;
            }
        }
    }

    pub fn deinit(self: *EmbeddingLayer) void {
        self.embedding_table.deinit();
    }
};

pub const MultiHeadSelfAttention = struct {
    num_heads: usize,
    embed_dim: usize,

    pub fn init(num_heads: usize, embed_dim: usize) !MultiHeadSelfAttention {
        if (num_heads == 0 or embed_dim % num_heads != 0) return error.ShapeMismatch;
        return .{ .num_heads = num_heads, .embed_dim = embed_dim };
    }

    pub fn forward(self: *MultiHeadSelfAttention, allocator: std.mem.Allocator, q: *trix.DataObject, k: *trix.DataObject, v: *trix.DataObject) !trix.DataObject {
        _ = self;
        const qs = q.shape.?.items;
        if (qs.len != 2) return error.ShapeMismatch;
        const seq = qs[0];
        const dim = qs[1];
        if (k.shape.?.items.len != 2 or v.shape.?.items.len != 2) return error.ShapeMismatch;
        if (k.shape.?.items[0] != seq or v.shape.?.items[0] != seq or k.shape.?.items[1] != dim or v.shape.?.items[1] != dim) return error.ShapeMismatch;
        var out = try trix.DataObject.init(allocator, &[_]usize{ seq, dim }, .f32);

        var scores = try trix.DataObject.init(allocator, &[_]usize{ seq, seq }, .f32);
        defer scores.deinit();
        const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dim)));
        for (0..seq) |i| {
            for (0..seq) |j| {
                var dot: f32 = 0.0;
                for (0..dim) |d| dot += q.values.items[i * dim + d] * k.values.items[j * dim + d];
                scores.values.items[i * seq + j] = dot * scale;
            }
            softmaxRow(scores.values.items[i * seq .. (i + 1) * seq]);
        }
        for (0..seq) |i| {
            for (0..dim) |d| {
                var acc: f32 = 0.0;
                for (0..seq) |j| {
                    acc += scores.values.items[i * seq + j] * v.values.items[j * dim + d];
                }
                out.values.items[i * dim + d] = acc;
            }
        }
        return out;
    }
};

pub fn residualAdd(allocator: std.mem.Allocator, x: *trix.DataObject, f_x: *trix.DataObject) !trix.DataObject {
    return core.add(allocator, x, f_x);
}

pub fn denseSkipConcat(allocator: std.mem.Allocator, tensors: []const *trix.DataObject) !trix.DataObject {
    if (tensors.len == 0) return error.ShapeMismatch;
    const batch = tensors[0].shape.?.items[0];
    var total_features: usize = 0;
    for (tensors) |t| {
        const s = t.shape.?.items;
        if (s.len != 2 or s[0] != batch) return error.ShapeMismatch;
        total_features += s[1];
    }
    var out = try trix.DataObject.init(allocator, &[_]usize{ batch, total_features }, .f32);
    for (0..batch) |b| {
        var cursor: usize = 0;
        for (tensors) |t| {
            const features = t.shape.?.items[1];
            const src = b * features;
            const dst = b * total_features + cursor;
            @memcpy(out.values.items[dst .. dst + features], t.values.items[src .. src + features]);
            cursor += features;
        }
    }
    return out;
}

pub const Sequential = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(LinearLayer),

    pub fn init(allocator: std.mem.Allocator) !Sequential {
        return .{ .allocator = allocator, .layers = try std.ArrayList(LinearLayer).initCapacity(allocator, 8) };
    }

    pub fn addLinear(self: *Sequential, input_size: usize, output_size: usize, activation: []const u8, forward_fn: ?ForwardFn, backprop_fn: ?BackpropFn) !void {
        try self.layers.append(try LinearLayer.init(self.allocator, input_size, output_size, activation, forward_fn, backprop_fn));
    }

    pub fn forward(self: *Sequential, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        var nn = NeuralNetwork{ .allocator = self.allocator, .layers = self.layers };
        return nn.forward(allocator, input);
    }

    pub fn deinit(self: *Sequential) void {
        for (self.layers.items) |*l| l.deinit();
        self.layers.deinit();
    }
};

pub fn branchForward(allocator: std.mem.Allocator, input: *trix.DataObject, branches: []const *LinearLayer) !trix.DataObject {
    if (branches.len == 0) return error.ShapeMismatch;
    const batch = input.shape.?.items[0];
    var total_features: usize = 0;
    for (branches) |b| {
        total_features += b.config.output_size;
    }
    var out = try trix.DataObject.init(allocator, &[_]usize{ batch, total_features }, .f32);
    for (0..batch) |b| {
        var cursor: usize = 0;
        for (branches) |br| {
            var branch_out = try br.forward(allocator, input);
            defer branch_out.deinit();
            const features = branch_out.shape.?.items[1];
            const src = b * features;
            const dst = b * total_features + cursor;
            @memcpy(out.values.items[dst .. dst + features], branch_out.values.items[src .. src + features]);
            cursor += features;
        }
    }
    return out;
}

pub const ResidualBlock = struct {
    main_1: LinearLayer,
    main_2: LinearLayer,
    skip: ?LinearLayer,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize, use_projection_skip: bool, forward_fn: ?ForwardFn, backprop_fn: ?BackpropFn) !ResidualBlock {
        return .{
            .main_1 = try LinearLayer.init(allocator, input_size, hidden_size, "relu", forward_fn, backprop_fn),
            .main_2 = try LinearLayer.init(allocator, hidden_size, output_size, "none", forward_fn, backprop_fn),
            .skip = if (use_projection_skip) try LinearLayer.init(allocator, input_size, output_size, "none", forward_fn, backprop_fn) else null,
            .allocator = allocator,
        };
    }

    pub fn forward(self: *ResidualBlock, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        var y = try self.main_1.forward(allocator, input);
        defer y.deinit();
        var f = try self.main_2.forward(allocator, &y);
        defer f.deinit();
        var skip_out = if (self.skip) |*s| try s.forward(allocator, input) else try cloneTensor(allocator, input);
        defer skip_out.deinit();
        return residualAdd(allocator, &skip_out, &f);
    }

    pub fn deinit(self: *ResidualBlock) void {
        self.main_1.deinit();
        self.main_2.deinit();
        if (self.skip) |*s| s.deinit();
    }
};

pub const InceptionModule = struct {
    branch_1x1: LinearLayer,
    branch_3x3: LinearLayer,
    branch_5x5: LinearLayer,
    pool_proj: LinearLayer,

    pub fn init(allocator: std.mem.Allocator, input: usize, c1: usize, c3: usize, c5: usize, cp: usize, forward_fn: ?ForwardFn, backprop_fn: ?BackpropFn) !InceptionModule {
        return .{
            .branch_1x1 = try LinearLayer.init(allocator, input, c1, "relu", forward_fn, backprop_fn),
            .branch_3x3 = try LinearLayer.init(allocator, input, c3, "relu", forward_fn, backprop_fn),
            .branch_5x5 = try LinearLayer.init(allocator, input, c5, "relu", forward_fn, backprop_fn),
            .pool_proj = try LinearLayer.init(allocator, input, cp, "relu", forward_fn, backprop_fn),
        };
    }

    pub fn forward(self: *InceptionModule, allocator: std.mem.Allocator, input: *trix.DataObject) !trix.DataObject {
        var a = try self.branch_1x1.forward(allocator, input);
        defer a.deinit();
        var b = try self.branch_3x3.forward(allocator, input);
        defer b.deinit();
        var c = try self.branch_5x5.forward(allocator, input);
        defer c.deinit();
        var d = try self.pool_proj.forward(allocator, input);
        defer d.deinit();
        return denseSkipConcat(allocator, &[_]*trix.DataObject{ &a, &b, &c, &d });
    }

    pub fn deinit(self: *InceptionModule) void {
        self.branch_1x1.deinit();
        self.branch_3x3.deinit();
        self.branch_5x5.deinit();
        self.pool_proj.deinit();
    }
};

pub const LayerStats = struct {
    name: []const u8,
    parameter_count: usize,
};

pub fn linearLayerStats(layer: *const LinearLayer) LayerStats {
    return .{
        .name = "LinearLayer",
        .parameter_count = layer.weights.values.items.len + layer.bias.values.items.len,
    };
}

pub fn conv2dLayerStats(layer: *const Conv2DLayer) LayerStats {
    return .{
        .name = "Conv2DLayer",
        .parameter_count = layer.weights.values.items.len + layer.bias.values.items.len,
    };
}

fn sigmoidScalar(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

fn softmaxRow(row: []f32) void {
    var maxv = row[0];
    for (row[1..]) |v| {
        if (v > maxv) maxv = v;
    }
    var sum_exp: f32 = 0.0;
    for (row) |*v| {
        v.* = std.math.exp(v.* - maxv);
        sum_exp += v.*;
    }
    for (row) |*v| v.* /= sum_exp;
}

fn cloneTensor(allocator: std.mem.Allocator, src: *trix.DataObject) !trix.DataObject {
    const out = try trix.DataObject.init(allocator, src.shape.?.items, .f32);
    @memcpy(out.values.items, src.values.items);
    return out;
}
