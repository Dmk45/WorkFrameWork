const std = @import("std");
const trix = @import("matrix.zig");
const grad_math = @import("grad_math.zig");

pub const OpRecord = struct {
    op: grad_math.OperationType,
    input_ids: std.ArrayList(usize),
    output_id: usize,
    metadata: grad_math.OperationMetadata,
};

pub const Tape = struct {
    allocator: std.mem.Allocator,
    tensors: std.AutoHashMap(usize, *trix.DataObject),
    ops: std.ArrayList(OpRecord),
    next_id: usize,

    pub fn init(allocator: std.mem.Allocator) Tape {
        return .{
            .allocator = allocator,
            .tensors = std.AutoHashMap(usize, *trix.DataObject).init(allocator),
            .ops = std.ArrayList(OpRecord).init(allocator),
            .next_id = 1,
        };
    }

    pub fn deinit(self: *Tape) void {
        for (self.ops.items) |*op| {
            op.input_ids.deinit();
        }
        self.ops.deinit();
        self.tensors.deinit();
    }

    pub fn registerTensor(self: *Tape, tensor: *trix.DataObject) !usize {
        const id = self.next_id;
        self.next_id += 1;
        try self.tensors.put(id, tensor);
        return id;
    }

    pub fn record(
        self: *Tape,
        op: grad_math.OperationType,
        input_ids: []const usize,
        output_id: usize,
        metadata: grad_math.OperationMetadata,
    ) !void {
        var ids = try std.ArrayList(usize).initCapacity(self.allocator, input_ids.len);
        for (input_ids) |id| try ids.append(id);
        try self.ops.append(.{
            .op = op,
            .input_ids = ids,
            .output_id = output_id,
            .metadata = metadata,
        });
    }

    pub fn backward(self: *Tape) !void {
        var i = self.ops.items.len;
        while (i > 0) {
            i -= 1;
            const rec = self.ops.items[i];
            const out = self.tensors.get(rec.output_id) orelse continue;
            var input_buf = try self.allocator.alloc(*trix.DataObject, rec.input_ids.items.len);
            defer self.allocator.free(input_buf);
            for (rec.input_ids.items, 0..) |id, idx| {
                input_buf[idx] = self.tensors.get(id) orelse continue;
            }
            try grad_math.executeBackward(rec.op, input_buf, out, rec.metadata, self.allocator);
        }
    }
};

pub fn stopGradient(tensor: *trix.DataObject) void {
    tensor.grad = false;
    if (tensor.grad_value) |*g| {
        @memset(g.items, 0.0);
    }
}

pub fn detach(allocator: std.mem.Allocator, tensor: *trix.DataObject) !trix.DataObject {
    const out = try trix.DataObject.init(allocator, tensor.shape.?.items, tensor.dtype);
    @memcpy(out.values.items, tensor.values.items);
    return out;
}

/// Gradient statistics for monitoring training
pub const GradientStats = struct {
    mean: f32,
    std: f32,
    max: f32,
    min: f32,
    norm: f32,

    pub fn compute(param: *trix.DataObject) ?GradientStats {
        if (param.grad_value == null) return null;

        const grads = param.grad_value.?.items;
        if (grads.len == 0) return null;

        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;
        var max_val: f32 = -std.math.inf(f32);
        var min_val: f32 = std.math.inf(f32);

        for (grads) |g| {
            sum += g;
            sum_sq += g * g;
            if (g > max_val) max_val = g;
            if (g < min_val) min_val = g;
        }

        const mean = sum / @as(f32, @floatFromInt(grads.len));
        const variance = (sum_sq / @as(f32, @floatFromInt(grads.len))) - (mean * mean);
        const std_val = if (variance > 0.0) std.math.sqrt(variance) else 0.0;
        const norm = std.math.sqrt(sum_sq);

        return GradientStats{
            .mean = mean,
            .std = std_val,
            .max = max_val,
            .min = min_val,
            .norm = norm,
        };
    }
};

/// Graph visualization utilities
pub const GraphVisualizer = struct {
    pub fn exportDot(self: *Tape, writer: anytype) !void {
        try writer.writeAll("digraph ComputationGraph {\n");
        try writer.writeAll("  rankdir=TB;\n");
        try writer.writeAll("  node [shape=box];\n\n");

        // Write tensor nodes
        var tensor_iter = self.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            const id = entry.key_ptr.*;
            const tensor = entry.value_ptr.*;
            const label = try std.fmt.allocPrint(self.allocator, "T{}\\nshape: {any}", .{ id, tensor.shape.? });
            defer self.allocator.free(label);
            try writer.print("  T{} [label=\"{}\"];\n", .{ id, label });
        }

        try writer.writeAll("\n");

        // Write operation nodes and edges
        for (self.ops.items) |op| {
            const op_name = @tagName(op.op);
            try writer.print("  O{} [label=\"{}\", shape=ellipse, style=filled, fillcolor=lightblue];\n", .{ op.output_id, op_name });

            // Edges from inputs to operation
            for (op.input_ids.items) |input_id| {
                try writer.print("  T{} -> O{};\n", .{ input_id, op.output_id });
            }

            // Edge from operation to output
            try writer.print("  O{} -> T{};\n", .{ op.output_id, op.output_id });
        }

        try writer.writeAll("}\n");
    }

    pub fn printSummary(self: *Tape) void {
        std.debug.print("Computation Graph Summary:\n");
        std.debug.print("  Tensors: {}\n", .{self.tensors.count()});
        std.debug.print("  Operations: {}\n", .{self.ops.items.len});
        std.debug.print("  Next ID: {}\n", .{self.next_id});
    }
};
