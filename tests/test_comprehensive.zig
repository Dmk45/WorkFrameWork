const std = @import("std");
const lib = @import("modelwork2");
const builder_mod = lib.maker;
const grad_mod = lib.grad;
const grad_math = lib.grad_math;
const core_mod = lib.core_math;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Test zeros for different dimensions
    std.debug.print("\n=== Testing Zeros ===\n", .{});

    var zeros_1d = try builder_mod.zeros(allocator, &[_]usize{ 4, 2 });
    defer zeros_1d.deinit();
    std.debug.print("1D Zeros: ", .{});
    zeros_1d.print();
    std.debug.print("\n", .{});

    // Test Mean Squared Error
    std.debug.print("\n=== Testing Mean Squared Error ===\n", .{});

    var y_pred = try builder_mod.zeros(allocator, &[_]usize{ 1, 4 });
    defer y_pred.deinit();
    y_pred.values.items[0] = 1.0;
    y_pred.values.items[1] = 2.0;
    y_pred.values.items[2] = 3.0;
    y_pred.values.items[3] = 4.0;
    y_pred.enableGrad();

    var y_true = try builder_mod.zeros(allocator, &[_]usize{ 1, 4 });
    defer y_true.deinit();
    y_true.values.items[0] = 1.5;
    y_true.values.items[1] = 2.5;
    y_true.values.items[2] = 3.5;
    y_true.values.items[3] = 4.5;

    const mse_loss = try grad_mod.meanSquaredError(&y_pred, &y_true);
    std.debug.print("MSE Loss: {d}\n", .{mse_loss});
    std.debug.print("Gradients: {any}\n", .{y_pred.grad_value.?.items});

    // Test Cross Entropy
    std.debug.print("\n=== Testing Cross Entropy ===\n", .{});

    var logits = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer logits.deinit();
    logits.values.items[0] = 1.0;
    logits.values.items[1] = 2.0;
    logits.values.items[2] = 0.5;
    logits.enableGrad();

    var targets = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer targets.deinit();
    targets.values.items[1] = 1.0; // one-hot for class 1

    const ce_loss = try grad_mod.crossEntropy(&logits, &targets, allocator);
    std.debug.print("Cross Entropy Loss: {d}\n", .{ce_loss});
    std.debug.print("Gradients: {any}\n", .{logits.grad_value.?.items});

    // Test Adam Optimizer
    std.debug.print("\n=== Testing Adam Optimizer ===\n", .{});

    var adam = try grad_mod.Adam.init(allocator, 0.01, 0.9, 0.999, 1e-8);
    defer adam.deinit();

    var param = try builder_mod.zeros(allocator, &[_]usize{ 1, 2 });
    defer param.deinit();
    param.values.items[0] = 1.0;
    param.values.items[1] = -1.0;
    param.enableGrad();

    // Simulate some gradients
    try param.ensureGradValue();
    param.grad_value.?.items[0] = 0.1;
    param.grad_value.?.items[1] = -0.2;

    std.debug.print("Before Adam step: {any}\n", .{param.values.items});
    try adam.step(&param);
    std.debug.print("After Adam step: {any}\n", .{param.values.items});

    // Test Automatic Backward Pass
    std.debug.print("\n=== Testing Automatic Backward Pass ===\n", .{});

    // Create input tensors
    var x = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer x.deinit();
    x.values.items[0] = 1.0;
    x.values.items[1] = 2.0;
    x.values.items[2] = 3.0;
    x.enableGrad();
    try x.ensureGradValue();

    var y = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer y.deinit();
    y.values.items[0] = 0.5;
    y.values.items[1] = 1.5;
    y.values.items[2] = 2.5;
    y.enableGrad();
    try y.ensureGradValue();

    // z = x + y
    var z = try core_mod.add(allocator, &x, &y);
    defer z.deinit();
    z.enableGrad();
    try z.ensureGradValue();

    // w = z * z (element-wise square)
    var w = try core_mod.mul(allocator, &z, &z);
    defer w.deinit();
    w.enableGrad();
    try w.ensureGradValue();

    // loss = sum(w)
    const loss_scalar = core_mod.sum(&w);
    var loss = try builder_mod.zeros(allocator, &[_]usize{ 1, 1 });
    defer loss.deinit();
    loss.values.items[0] = loss_scalar;

    std.debug.print("x: {any}\n", .{x.values.items});
    std.debug.print("y: {any}\n", .{y.values.items});
    std.debug.print("z (x+y): {any}\n", .{z.values.items});
    std.debug.print("w (z*z): {any}\n", .{w.values.items});
    std.debug.print("loss (sum(w)): {d}\n", .{loss.values.items[0]});

    // Manual backward pass using executeBackward
    std.debug.print("\n--- Manual Backward Pass ---\n", .{});

    // dL/dw = 1 (scalar gradient from sum)
    var d_loss = try builder_mod.zeros(allocator, &[_]usize{ 1, 1 });
    defer d_loss.deinit();
    d_loss.values.items[0] = 1.0;

    // Backward through sum: dL/dw = broadcast(dL/dloss)
    try grad_math.sumBackward(&w, &d_loss);
    std.debug.print("dL/dw: {any}\n", .{w.grad_value.?.items});

    // Create a tensor with w's gradients for the next backward call
    var d_w = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer d_w.deinit();
    for (0..3) |i| {
        d_w.values.items[i] = w.grad_value.?.items[i];
    }

    // Backward through mul: dL/dz += dL/dw * z + dL/dw * z
    try grad_math.mulBackward(&z, &z, &d_w);
    std.debug.print("dL/dz: {any}\n", .{z.grad_value.?.items});

    // Create a tensor with z's gradients
    var d_z = try builder_mod.zeros(allocator, &[_]usize{ 1, 3 });
    defer d_z.deinit();
    for (0..3) |i| {
        d_z.values.items[i] = z.grad_value.?.items[i];
    }

    // Backward through add: dL/dx += dL/dz, dL/dy += dL/dz
    try grad_math.addBackward(&x, &y, &d_z);
    std.debug.print("dL/dx: {any}\n", .{x.grad_value.?.items});
    std.debug.print("dL/dy: {any}\n", .{y.grad_value.?.items});

    // Verify gradients analytically
    // loss = sum((x+y)^2) = sum(x^2 + 2xy + y^2)
    // dL/dx = 2(x+y), dL/dy = 2(x+y)
    // For x=[1,2,3], y=[0.5,1.5,2.5], z=[1.5,3.5,5.5]
    // dL/dx should be [3,7,11], dL/dy should be [3,7,11]
    std.debug.print("Expected dL/dx: [3, 7, 11]\n", .{});
    std.debug.print("Expected dL/dy: [3, 7, 11]\n", .{});
}
