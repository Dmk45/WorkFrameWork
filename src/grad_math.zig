const std = @import("std");
const trix = @import("matrix.zig");
const core = @import("core_math.zig");

// =============================================================================
// TENSOR REGISTRY FOR FFI
// =============================================================================

var tensor_registry: std.AutoHashMap(usize, *trix.DataObject) = undefined;

/// Initialize the tensor registry
pub fn grad_math_init_registry(allocator: std.mem.Allocator) void {
    tensor_registry = std.AutoHashMap(usize, *trix.DataObject).init(allocator);
}

/// Register a tensor in the registry
export fn grad_math_register_tensor(id: usize, data: *trix.DataObject) void {
    tensor_registry.put(id, data) catch {};
}

/// Get a tensor from the registry
fn get_tensor(id: usize) ?*trix.DataObject {
    return tensor_registry.get(id);
}

/// Gradient computation for element-wise addition: d(a) += d(output), d(b) += d(output)
/// Both inputs must have gradients enabled
pub fn addBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (a.grad_value == null or b.grad_value == null) return;

    for (0..d_output.values.items.len) |i| {
        a.grad_value.?.items[i] += d_output.values.items[i];
        b.grad_value.?.items[i] += d_output.values.items[i];
    }
}

/// Gradient computation for element-wise subtraction: d(a) += d(output), d(b) -= d(output)
pub fn subBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (a.grad_value == null or b.grad_value == null) return;

    for (0..d_output.values.items.len) |i| {
        a.grad_value.?.items[i] += d_output.values.items[i];
        b.grad_value.?.items[i] -= d_output.values.items[i];
    }
}

/// Gradient computation for element-wise multiplication: d(a) += d(output) * b, d(b) += d(output) * a
pub fn mulBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (a.grad_value == null or b.grad_value == null) return;

    for (0..d_output.values.items.len) |i| {
        a.grad_value.?.items[i] += d_output.values.items[i] * b.values.items[i];
        b.grad_value.?.items[i] += d_output.values.items[i] * a.values.items[i];
    }
}

/// Gradient computation for element-wise division: d(a) += d(output) / b, d(b) -= d(output) * a / (b^2)
pub fn divBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (a.grad_value == null or b.grad_value == null) return;

    for (0..d_output.values.items.len) |i| {
        const b_val = b.values.items[i];
        if (b_val != 0.0) {
            a.grad_value.?.items[i] += d_output.values.items[i] / b_val;
            b.grad_value.?.items[i] -= d_output.values.items[i] * a.values.items[i] / (b_val * b_val);
        }
    }
}

/// Gradient computation for scaling: d(tensor) += d(output) * scalar
pub fn scaleBackward(tensor: *trix.DataObject, scalar: f32, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    for (0..d_output.values.items.len) |i| {
        tensor.grad_value.?.items[i] += d_output.values.items[i] * scalar;
    }
}

/// Gradient computation for adding scalar: d(tensor) += sum(d(output))
pub fn addScalarBackward(tensor: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    var sum_grad: f32 = 0.0;
    for (d_output.values.items) |grad| {
        sum_grad += grad;
    }

    for (0..tensor.grad_value.?.items.len) |i| {
        tensor.grad_value.?.items[i] += sum_grad;
    }
}

/// Gradient computation for matrix multiplication
/// For C = A @ B, we have:
/// d(A) += d(C) @ B^T
/// Gradient computation for matrix multiplication
/// For C = A @ B, we have:
/// d(A) += d(C) @ B^T
/// d(B) += A^T @ d(C)
pub fn matmulBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject, allocator: std.mem.Allocator) !void {
    const a_shape = a.shape.?.items;
    const b_shape = b.shape.?.items;
    const d_shape = d_output.shape.?.items;

    const is_batched = a_shape.len == 3 and b_shape.len == 3 and d_shape.len == 3;

    if (is_batched) {
        // Batched case: [B, M, K] @ [B, K, N] -> [B, M, N]
        if (a_shape.len != 3 or b_shape.len != 3) {
            return error.ShapeMismatch;
        }

        if (b_shape[0] != a_shape[0] or b_shape[1] != a_shape[2]) {
            return error.ShapeMismatch;
        }

        if (a.grad_value != null) {
            // d(A) += d(C) @ B^T
            var b_t = try core.transpose(allocator, b);
            defer b_t.deinit();

            var d_a = try core.matmul(allocator, d_output, &b_t);
            defer d_a.deinit();

            for (0..d_a.values.items.len) |i| {
                a.grad_value.?.items[i] += d_a.values.items[i];
            }
        }

        if (b.grad_value != null) {
            // d(B) += A^T @ d(C)
            var a_t = try core.transpose(allocator, a);
            defer a_t.deinit();

            var d_b = try core.matmul(allocator, &a_t, d_output);
            defer d_b.deinit();

            for (0..d_b.values.items.len) |i| {
                b.grad_value.?.items[i] += d_b.values.items[i];
            }
        }
    } else {
        // Standard 2D case: [M, K] @ [K, N] -> [M, N]
        if (a_shape.len != 2 or b_shape.len != 2) {
            return error.ShapeMismatch;
        }

        if (b_shape[0] != a_shape[1]) {
            return error.ShapeMismatch;
        }

        if (a.grad_value != null) {
            // d(A) += d(C) @ B^T
            var b_t = try core.transpose(allocator, b);
            defer b_t.deinit();

            var d_a = try core.matmul(allocator, d_output, &b_t);
            defer d_a.deinit();

            for (0..d_a.values.items.len) |i| {
                a.grad_value.?.items[i] += d_a.values.items[i];
            }
        }

        if (b.grad_value != null) {
            // d(B) += A^T @ d(C)
            var a_t = try core.transpose(allocator, a);
            defer a_t.deinit();

            var d_b = try core.matmul(allocator, &a_t, d_output);
            defer d_b.deinit();

            for (0..d_b.values.items.len) |i| {
                b.grad_value.?.items[i] += d_b.values.items[i];
            }
        }
    }
}

/// Gradient computation for transpose: d(input) += transpose(d(output))
pub fn transposeBackward(input: *trix.DataObject, d_output: *trix.DataObject, allocator: std.mem.Allocator) !void {
    if (input.grad_value == null) return;

    // Transpose the gradient back to input shape
    var d_input = try core.transpose(allocator, d_output);
    defer d_input.deinit();

    for (0..d_input.values.items.len) |i| {
        input.grad_value.?.items[i] += d_input.values.items[i];
    }
}

/// Gradient computation for reshape: d(input) += reshape(d(output))
pub fn reshapeBackward(input: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    // Reshape gradient back to input shape
    try core.reshape(d_output, input.shape.?.items);

    for (0..d_output.values.items.len) |i| {
        input.grad_value.?.items[i] += d_output.values.items[i];
    }
}

/// Gradient computation for sum: d(tensor) += broadcast(d(output))
pub fn sumBackward(tensor: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    // Broadcast scalar gradient to all elements
    const grad_val = d_output.values.items[0];
    for (0..tensor.grad_value.?.items.len) |i| {
        tensor.grad_value.?.items[i] += grad_val;
    }
}

/// Gradient computation for mean: d(tensor) += broadcast(d(output) / n)
pub fn meanBackward(tensor: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    // Broadcast scalar gradient divided by tensor size
    const n = @as(f32, @floatFromInt(tensor.values.items.len));
    const grad_val = d_output.values.items[0] / n;

    for (0..tensor.grad_value.?.items.len) |i| {
        tensor.grad_value.?.items[i] += grad_val;
    }
}

/// Gradient computation for dot product: d(a) += d(output) * b, d(b) += d(output) * a
pub fn dotBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (a.grad_value == null or b.grad_value == null) return;

    const grad_val = d_output.values.items[0];

    for (0..a.values.items.len) |i| {
        a.grad_value.?.items[i] += grad_val * b.values.items[i];
        b.grad_value.?.items[i] += grad_val * a.values.items[i];
    }
}

/// Gradient computation for bias addition: d(matrix) += d(output), d(bias) += sum(d(output), axis=0)
pub fn addBiasBackward(matrix: *trix.DataObject, bias: *trix.DataObject, d_output: *trix.DataObject) !void {
    const shape = matrix.shape.?.items;

    if (matrix.grad_value != null) {
        // d(matrix) += d(output)
        for (0..d_output.values.items.len) |i| {
            matrix.grad_value.?.items[i] += d_output.values.items[i];
        }
    }

    if (bias.grad_value != null) {
        // d(bias) += sum over first dimensions
        if (shape.len == 2) {
            // 2D case: sum over rows
            const m = shape[0];
            const n = shape[1];

            for (0..n) |j| {
                var sum_grad: f32 = 0.0;
                for (0..m) |i| {
                    sum_grad += d_output.values.items[i * n + j];
                }
                bias.grad_value.?.items[j] += sum_grad;
            }
        } else if (shape.len == 3) {
            // 3D case: sum over batch and rows
            const b = shape[0];
            const m = shape[1];
            const n = shape[2];

            for (0..n) |j| {
                var sum_grad: f32 = 0.0;
                for (0..b) |batch| {
                    for (0..m) |i| {
                        const idx = batch * m * n + i * n + j;
                        sum_grad += d_output.values.items[idx];
                    }
                }
                bias.grad_value.?.items[j] += sum_grad;
            }
        }
    }
}

/// Gradient computation for matrix-vector multiplication: d(matrix) += outer(d(output), vec), d(vec) += matrix^T @ d(output)
pub fn matvecBackward(matrix: *trix.DataObject, vec: *trix.DataObject, d_output: *trix.DataObject, allocator: std.mem.Allocator) !void {
    if (matrix.grad_value != null) {
        // d(matrix) += outer(d(output), vec)
        var outer_prod = try core.outer(allocator, d_output, vec);
        defer outer_prod.deinit();

        for (0..outer_prod.values.items.len) |i| {
            matrix.grad_value.?.items[i] += outer_prod.values.items[i];
        }
    }

    if (vec.grad_value != null) {
        // d(vec) += matrix^T @ d(output)
        var matrix_t = try core.transpose(allocator, matrix);
        defer matrix_t.deinit();

        var d_vec = try core.matvec(allocator, &matrix_t, d_output);
        defer d_vec.deinit();

        for (0..d_vec.values.items.len) |i| {
            vec.grad_value.?.items[i] += d_vec.values.items[i];
        }
    }
}

/// Gradient computation for outer product: d(a) += sum(d(output), axis=1), d(b) += sum(d(output), axis=0)
pub fn outerBackward(a: *trix.DataObject, b: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (a.grad_value != null) {
        // d(a) += sum(d(output), axis=1)
        const n = b.values.items.len;
        for (0..a.values.items.len) |i| {
            var sum_grad: f32 = 0.0;
            for (0..n) |j| {
                sum_grad += d_output.values.items[i * n + j];
            }
            a.grad_value.?.items[i] += sum_grad;
        }
    }

    if (b.grad_value != null) {
        // d(b) += sum(d(output), axis=0)
        const m = a.values.items.len;
        for (0..b.values.items.len) |j| {
            var sum_grad: f32 = 0.0;
            for (0..m) |i| {
                sum_grad += d_output.values.items[i * b.values.items.len + j];
            }
            b.grad_value.?.items[j] += sum_grad;
        }
    }
}

/// Gradient computation for Frobenius norm: d(tensor) += d(output) * tensor / norm
pub fn frobeniusNormBackward(tensor: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    const norm = core.frobeniusNorm(tensor);
    if (norm == 0.0) return;

    const grad_val = d_output.values.items[0];
    for (0..tensor.values.items.len) |i| {
        tensor.grad_value.?.items[i] += grad_val * tensor.values.items[i] / norm;
    }
}

/// Gradient computation for clamping: d(tensor) += d(output) where tensor is within bounds
pub fn clampBackward(tensor: *trix.DataObject, min_val: f32, max_val: f32, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    for (0..tensor.values.items.len) |i| {
        const val = tensor.values.items[i];
        if (val >= min_val and val <= max_val) {
            tensor.grad_value.?.items[i] += d_output.values.items[i];
        }
        // Otherwise gradient is 0 (blocked by clamping)
    }
}

/// Gradient computation for absolute value: d(tensor) += d(output) * sign(tensor)
pub fn absBackward(tensor: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (tensor.grad_value == null) return;

    for (0..tensor.values.items.len) |i| {
        const sign: f32 = if (tensor.values.items[i] > 0.0) 1.0 else if (tensor.values.items[i] < 0.0) -1.0 else 0.0;
        tensor.grad_value.?.items[i] += d_output.values.items[i] * sign;
    }
}

// =============================================================================
// ACTIVATION FUNCTION GRADIENTS
// =============================================================================

/// Gradient computation for ReLU: d(input) += d(output) where input > 0, else 0
pub fn reluBackward(input: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    for (0..input.values.items.len) |i| {
        if (input.values.items[i] > 0.0) {
            input.grad_value.?.items[i] += d_output.values.items[i];
        }
        // Otherwise gradient is 0 (ReLU blocks negative gradients)
    }
}

/// Gradient computation for Sigmoid: d(input) += d(output) * sigmoid(input) * (1 - sigmoid(input))
pub fn sigmoidBackward(input: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    for (0..input.values.items.len) |i| {
        const x = input.values.items[i];
        const sigmoid_x = 1.0 / (1.0 + std.math.exp(-x));
        const grad = sigmoid_x * (1.0 - sigmoid_x);
        input.grad_value.?.items[i] += d_output.values.items[i] * grad;
    }
}

/// Gradient computation for Tanh: d(input) += d(output) * (1 - tanh(input)^2)
pub fn tanhBackward(input: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    for (0..input.values.items.len) |i| {
        const x = input.values.items[i];
        const tanh_x = std.math.tanh(x);
        const grad = 1.0 - tanh_x * tanh_x;
        input.grad_value.?.items[i] += d_output.values.items[i] * grad;
    }
}

/// Gradient computation for ELU (Exponential Linear Unit): d(input) += d(output) * d/dx(ELU(x))
pub fn eluBackward(input: *trix.DataObject, alpha: f32, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    for (0..input.values.items.len) |i| {
        const x = input.values.items[i];
        const grad = if (x > 0.0) 1.0 else alpha * std.math.exp(x);
        input.grad_value.?.items[i] += d_output.values.items[i] * grad;
    }
}

/// Gradient computation for LeakyReLU: d(input) += d(output) * (1 if x > 0 else negative_slope)
pub fn leakyReluBackward(input: *trix.DataObject, negative_slope: f32, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    for (0..input.values.items.len) |i| {
        const grad = if (input.values.items[i] > 0.0) 1.0 else negative_slope;
        input.grad_value.?.items[i] += d_output.values.items[i] * grad;
    }
}

/// Gradient computation for GELU (Gaussian Error Linear Unit)
/// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
/// d/dx(GELU(x)) = Φ(x) + x * φ(x) where φ is the PDF
pub fn geluBackward(input: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    for (0..input.values.items.len) |i| {
        const x = input.values.items[i];
        // Approximation of CDF: 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const sqrt_2_pi = std.math.sqrt(2.0 / std.math.pi);
        const inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        const cdf = 0.5 * (1.0 + std.math.tanh(inner));

        // PDF: (1/sqrt(2π)) * exp(-x^2/2)
        const pdf = (1.0 / std.math.sqrt(2.0 * std.math.pi)) * std.math.exp(-x * x / 2.0);

        const grad = cdf + x * pdf;
        input.grad_value.?.items[i] += d_output.values.items[i] * grad;
    }
}

/// Gradient computation for Softmax (vectorized)
/// For softmax output s, d(s)/d(input) is complex - requires storing the softmax output
/// This assumes we have the softmax output stored and computes the Jacobian-vector product
pub fn softmaxBackward(input: *trix.DataObject, softmax_output: *trix.DataObject, d_output: *trix.DataObject) !void {
    if (input.grad_value == null) return;

    const shape = softmax_output.shape.?.items;
    if (shape.len != 2) return; // Only support 2D softmax for now

    const batch_size = shape[0];
    const num_classes = shape[1];

    for (0..batch_size) |b| {
        // For each sample in batch, compute Jacobian-vector product
        for (0..num_classes) |i| {
            var sum: f32 = 0.0;
            for (0..num_classes) |j| {
                const s_i = softmax_output.values.items[b * num_classes + i];
                const s_j = softmax_output.values.items[b * num_classes + j];
                const d_out_j = d_output.values.items[b * num_classes + j];

                if (i == j) {
                    sum += d_out_j * s_i * (1.0 - s_j);
                } else {
                    sum -= d_out_j * s_i * s_j;
                }
            }
            input.grad_value.?.items[b * num_classes + i] += sum;
        }
    }
}

// =============================================================================
// BACKWARD PASS LOGIC & FFI INTERFACE
// =============================================================================

/// Operation types that can be executed in backward pass
pub const OperationType = enum {
    Add,
    Sub,
    Mul,
    Div,
    Scale,
    AddScalar,
    Transpose,
    Reshape,
    Sum,
    Mean,
    Dot,
    AddBias,
    MatVec,
    Outer,
    FrobeniusNorm,
    Clamp,
    Abs,
    ReLU,
    Sigmoid,
    Tanh,
    ELU,
    LeakyReLU,
    GELU,
    Softmax,
    MatMul,
};

/// Metadata for operations that need additional parameters
pub const OperationMetadata = union(OperationType) {
    Add: void,
    Sub: void,
    Mul: void,
    Div: void,
    Scale: f32,
    AddScalar: f32,
    Transpose: void,
    Reshape: void,
    Sum: void,
    Mean: void,
    Dot: void,
    AddBias: void,
    MatVec: void,
    Outer: void,
    FrobeniusNorm: void,
    Clamp: struct { min: f32, max: f32 },
    Abs: void,
    ReLU: void,
    Sigmoid: void,
    Tanh: void,
    ELU: f32,
    LeakyReLU: f32,
    GELU: void,
    Softmax: []const usize, // output shape
    MatMul: void,
};

/// Execute backward pass for a single operation
/// This is the main dispatcher that calls the appropriate backward function
pub fn executeBackward(
    op: OperationType,
    inputs: []const *trix.DataObject,
    output: *trix.DataObject,
    metadata: OperationMetadata,
    allocator: std.mem.Allocator,
) !void {
    switch (op) {
        .Add => {
            if (inputs.len >= 2) {
                try addBackward(inputs[0], inputs[1], output);
            }
        },
        .Sub => {
            if (inputs.len >= 2) {
                try subBackward(inputs[0], inputs[1], output);
            }
        },
        .Mul => {
            if (inputs.len >= 2) {
                try mulBackward(inputs[0], inputs[1], output);
            }
        },
        .Div => {
            if (inputs.len >= 2) {
                try divBackward(inputs[0], inputs[1], output);
            }
        },
        .Scale => {
            if (inputs.len >= 1) {
                switch (metadata) {
                    .Scale => |scalar| try scaleBackward(inputs[0], scalar, output),
                    else => {},
                }
            }
        },
        .AddScalar => {
            if (inputs.len >= 1) {
                try addScalarBackward(inputs[0], output);
            }
        },
        .Transpose => {
            if (inputs.len >= 1) {
                try transposeBackward(inputs[0], output, allocator);
            }
        },
        .Reshape => {
            if (inputs.len >= 1) {
                try reshapeBackward(inputs[0], output);
            }
        },
        .Sum => {
            if (inputs.len >= 1) {
                try sumBackward(inputs[0], output);
            }
        },
        .Mean => {
            if (inputs.len >= 1) {
                try meanBackward(inputs[0], output);
            }
        },
        .Dot => {
            if (inputs.len >= 2) {
                try dotBackward(inputs[0], inputs[1], output);
            }
        },
        .AddBias => {
            if (inputs.len >= 2) {
                try addBiasBackward(inputs[0], inputs[1], output);
            }
        },
        .MatVec => {
            if (inputs.len >= 2) {
                try matvecBackward(inputs[0], inputs[1], output, allocator);
            }
        },
        .Outer => {
            if (inputs.len >= 2) {
                try outerBackward(inputs[0], inputs[1], output);
            }
        },
        .FrobeniusNorm => {
            if (inputs.len >= 1) {
                try frobeniusNormBackward(inputs[0], output);
            }
        },
        .Clamp => {
            if (inputs.len >= 1) {
                switch (metadata) {
                    .Clamp => |bounds| try clampBackward(inputs[0], bounds.min, bounds.max, output),
                    else => {},
                }
            }
        },
        .Abs => {
            if (inputs.len >= 1) {
                try absBackward(inputs[0], output);
            }
        },
        .ReLU => {
            if (inputs.len >= 1) {
                try reluBackward(inputs[0], output);
            }
        },
        .Sigmoid => {
            if (inputs.len >= 1) {
                try sigmoidBackward(inputs[0], output);
            }
        },
        .Tanh => {
            if (inputs.len >= 1) {
                try tanhBackward(inputs[0], output);
            }
        },
        .ELU => {
            if (inputs.len >= 1) {
                switch (metadata) {
                    .ELU => |alpha| try eluBackward(inputs[0], alpha, output),
                    else => {},
                }
            }
        },
        .LeakyReLU => {
            if (inputs.len >= 1) {
                switch (metadata) {
                    .LeakyReLU => |slope| try leakyReluBackward(inputs[0], slope, output),
                    else => {},
                }
            }
        },
        .GELU => {
            if (inputs.len >= 1) {
                try geluBackward(inputs[0], output);
            }
        },
        .Softmax => {
            if (inputs.len >= 1) {
                // For softmax, we need the output tensor as an additional input
                // This assumes the output is stored and passed as the first input
                try softmaxBackward(inputs[0], output, output);
            }
        },
        .MatMul => {
            if (inputs.len >= 2) {
                try matmulBackward(inputs[0], inputs[1], output, allocator);
            }
        },
    }
}

// =============================================================================
// FFI EXPORTS FOR RUST INTEROP
// =============================================================================

/// FFI export: Execute backward pass for addition
export fn grad_math_add_backward(a_id: usize, b_id: usize, d_output_id: usize) void {
    const a = get_tensor(a_id) orelse return;
    const b = get_tensor(b_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    addBackward(a, b, d_output) catch {};
}

/// FFI export: Execute backward pass for subtraction
export fn grad_math_sub_backward(a_id: usize, b_id: usize, d_output_id: usize) void {
    const a = get_tensor(a_id) orelse return;
    const b = get_tensor(b_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    subBackward(a, b, d_output) catch {};
}

/// FFI export: Execute backward pass for multiplication
export fn grad_math_mul_backward(a_id: usize, b_id: usize, d_output_id: usize) void {
    const a = get_tensor(a_id) orelse return;
    const b = get_tensor(b_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    mulBackward(a, b, d_output) catch {};
}

/// FFI export: Execute backward pass for division
export fn grad_math_div_backward(a_id: usize, b_id: usize, d_output_id: usize) void {
    const a = get_tensor(a_id) orelse return;
    const b = get_tensor(b_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    divBackward(a, b, d_output) catch {};
}

/// FFI export: Execute backward pass for scaling
export fn grad_math_scale_backward(tensor_id: usize, scalar: f32, d_output_id: usize) void {
    const tensor = get_tensor(tensor_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    scaleBackward(tensor, scalar, d_output) catch {};
}

/// FFI export: Execute backward pass for matrix multiplication
export fn grad_math_matmul_backward(a_id: usize, b_id: usize, d_output_id: usize, allocator: *std.mem.Allocator) void {
    const a = get_tensor(a_id) orelse return;
    const b = get_tensor(b_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    matmulBackward(a, b, d_output, allocator.*) catch {};
}

/// FFI export: Execute backward pass for ReLU
export fn grad_math_relu_backward(input_id: usize, d_output_id: usize) void {
    const input = get_tensor(input_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    reluBackward(input, d_output) catch {};
}

/// FFI export: Execute backward pass for Sigmoid
export fn grad_math_sigmoid_backward(input_id: usize, d_output_id: usize) void {
    const input = get_tensor(input_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    sigmoidBackward(input, d_output) catch {};
}

/// FFI export: Execute backward pass for Tanh
export fn grad_math_tanh_backward(input_id: usize, d_output_id: usize) void {
    const input = get_tensor(input_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    tanhBackward(input, d_output) catch {};
}

/// FFI export: Execute backward pass for Softmax
export fn grad_math_softmax_backward(input_id: usize, softmax_output_id: usize, d_output_id: usize) void {
    const input = get_tensor(input_id) orelse return;
    const softmax_output = get_tensor(softmax_output_id) orelse return;
    const d_output = get_tensor(d_output_id) orelse return;
    softmaxBackward(input, softmax_output, d_output) catch {};
}
