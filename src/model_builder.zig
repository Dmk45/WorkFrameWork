const std = @import("std");
const trix = @import("matrix.zig");
const layers = @import("layers.zig");
const grad = @import("grad.zig");

/// Dynamic shape support for variable input dimensions
pub const DynamicShape = struct {
    min_dims: []const usize,
    max_dims: []const usize,
    current_dims: []usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, min_dims: []const usize, max_dims: []const usize) !DynamicShape {
        if (min_dims.len != max_dims.len) return error.DimensionMismatch;
        
        var current_dims = try allocator.alloc(usize, min_dims.len);
        for (min_dims, 0..) |dim, i| {
            current_dims[i] = dim;
        }
        
        return .{
            .min_dims = min_dims,
            .max_dims = max_dims,
            .current_dims = current_dims,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DynamicShape) void {
        self.allocator.free(self.current_dims);
    }
    
    pub fn resize(self: *DynamicShape, new_dims: []const usize) !void {
        if (new_dims.len != self.min_dims.len) return error.DimensionMismatch;
        
        for (new_dims, self.min_dims, self.max_dims, 0..) |new, min, max, i| {
            if (new < min or new > max) return error.OutOfRange;
            self.current_dims[i] = new;
        }
    }
    
    pub fn getShape(self: *DynamicShape) []const usize {
        return self.current_dims;
    }
};

/// Model configuration for parameterized models
pub const ModelConfig = struct {
    name: []const u8,
    input_size: usize,
    hidden_sizes: []const usize,
    output_size: usize,
    activations: []const []const u8,
    dropout_rates: ?[]const f32,
    use_batch_norm: bool,
    use_layer_norm: bool,
    
    pub fn fromJson(allocator: std.mem.Allocator, json_str: []const u8) !ModelConfig {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
        defer parsed.deinit();
        
        const obj = parsed.value.object;
        
        var hidden_sizes = std.ArrayList(usize).init(allocator);
        var activations = std.ArrayList([]const u8).init(allocator);
        var dropout_rates = std.ArrayList(f32).init(allocator);
        
        if (obj.get("hidden_sizes")) |hs_array| {
            for (hs_array.array.items) |hs| {
                try hidden_sizes.append(@intCast(hs.integer));
            }
        }
        
        if (obj.get("activations")) |act_array| {
            for (act_array.array.items) |act| {
                try activations.append(try allocator.dupe(u8, act.string));
            }
        }
        
        if (obj.get("dropout_rates")) |dr_array| {
            for (dr_array.array.items) |dr| {
                try dropout_rates.append(@floatCast(dr.float));
            }
        }
        
        return ModelConfig{
            .name = try allocator.dupe(u8, obj.get("name").?.string),
            .input_size = @intCast(obj.get("input_size").?.integer),
            .hidden_sizes = hidden_sizes.toOwnedSlice(),
            .output_size = @intCast(obj.get("output_size").?.integer),
            .activations = activations.toOwnedSlice(),
            .dropout_rates = if (dropout_rates.items.len > 0) dropout_rates.toOwnedSlice() else null,
            .use_batch_norm = obj.get("use_batch_norm").?.bool,
            .use_layer_norm = obj.get("use_layer_norm").?.bool,
        };
    }
    
    pub fn toJson(self: *ModelConfig, allocator: std.mem.Allocator) ![]const u8 {
        var obj = std.json.ObjectMap.init(allocator);
        defer obj.deinit();
        
        try obj.put("name", std.json.Value{ .string = self.name });
        try obj.put("input_size", std.json.Value{ .integer = @intCast(self.input_size) });
        try obj.put("output_size", std.json.Value{ .integer = @intCast(self.output_size) });
        try obj.put("use_batch_norm", std.json.Value{ .bool = self.use_batch_norm });
        try obj.put("use_layer_norm", std.json.Value{ .bool = self.use_layer_norm });
        
        var hidden_sizes_array = std.ArrayList(std.json.Value).init(allocator);
        for (self.hidden_sizes) |hs| {
            try hidden_sizes_array.append(std.json.Value{ .integer = @intCast(hs) });
        }
        try obj.put("hidden_sizes", std.json.Value{ .array = hidden_sizes_array.toOwnedSlice() });
        
        var activations_array = std.ArrayList(std.json.Value).init(allocator);
        for (self.activations) |act| {
            try activations_array.append(std.json.Value{ .string = act });
        }
        try obj.put("activations", std.json.Value{ .array = activations_array.toOwnedSlice() });
        
        if (self.dropout_rates) |dr| {
            var dr_array = std.ArrayList(std.json.Value).init(allocator);
            for (dr) |rate| {
                try dr_array.append(std.json.Value{ .float = rate });
            }
            try obj.put("dropout_rates", std.json.Value{ .array = dr_array.toOwnedSlice() });
        }
        
        const json_str = try std.json.stringifyAlloc(allocator, obj, .{ .whitespace = .indent_2 });
        return json_str;
    }
};

/// Model templates for common architectures
pub const ModelTemplates = struct {
    pub fn createResNet(allocator: std.mem.Allocator, variant: []const u8, num_classes: usize) !layers.NeuralNetwork {
        if (std.mem.eql(u8, variant, "resnet18")) {
            return createResNet18(allocator, num_classes);
        } else if (std.mem.eql(u8, variant, "resnet34")) {
            return createResNet34(allocator, num_classes);
        } else if (std.mem.eql(u8, variant, "resnet50")) {
            return createResNet50(allocator, num_classes);
        } else {
            return error.UnsupportedVariant;
        }
    }
    
    fn createResNet18(allocator: std.mem.Allocator, num_classes: usize) !layers.NeuralNetwork {
        var nn = layers.NeuralNetwork.init(allocator);
        // Simplified ResNet-18 structure
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 3, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, num_classes, "none" });
        return nn;
    }
    
    fn createResNet34(allocator: std.mem.Allocator, num_classes: usize) !layers.NeuralNetwork {
        var nn = layers.NeuralNetwork.init(allocator);
        // Simplified ResNet-34 structure
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 3, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 1024, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 1024, num_classes, "none" });
        return nn;
    }
    
    fn createResNet50(allocator: std.mem.Allocator, num_classes: usize) !layers.NeuralNetwork {
        var nn = layers.NeuralNetwork.init(allocator);
        // Simplified ResNet-50 structure
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 3, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 1024, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 1024, 2048, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 2048, num_classes, "none" });
        return nn;
    }
    
    pub fn createVGG(allocator: std.mem.Allocator, variant: []const u8, num_classes: usize) !layers.NeuralNetwork {
        if (std.mem.eql(u8, variant, "vgg11")) {
            return createVGG11(allocator, num_classes);
        } else if (std.mem.eql(u8, variant, "vgg16")) {
            return createVGG16(allocator, num_classes);
        } else if (std.mem.eql(u8, variant, "vgg19")) {
            return createVGG19(allocator, num_classes);
        } else {
            return error.UnsupportedVariant;
        }
    }
    
    fn createVGG11(allocator: std.mem.Allocator, num_classes: usize) !layers.NeuralNetwork {
        var nn = layers.NeuralNetwork.init(allocator);
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 3, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, num_classes, "none" });
        return nn;
    }
    
    fn createVGG16(allocator: std.mem.Allocator, num_classes: usize) !layers.NeuralNetwork {
        var nn = layers.NeuralNetwork.init(allocator);
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 3, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, num_classes, "none" });
        return nn;
    }
    
    fn createVGG19(allocator: std.mem.Allocator, num_classes: usize) !layers.NeuralNetwork {
        var nn = layers.NeuralNetwork.init(allocator);
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 3, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 64, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 64, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 128, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 128, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 256, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 256, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, 512, "relu" });
        try layers.Layer.addToNN(layers.LinearLayer, &nn, .{ 512, num_classes, "none" });
        return nn;
    }
};

/// Custom layer registration system
pub const CustomLayerRegistry = struct {
    layers: std.StringHashMap(*anyopaque),
    forward_fns: std.StringHashMap(*anyopaque),
    backward_fns: std.StringHashMap(*anyopaque),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CustomLayerRegistry {
        return .{
            .layers = std.StringHashMap(*anyopaque).init(allocator),
            .forward_fns = std.StringHashMap(*anyopaque).init(allocator),
            .backward_fns = std.StringHashMap(*anyopaque).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *CustomLayerRegistry) void {
        self.layers.deinit();
        self.forward_fns.deinit();
        self.backward_fns.deinit();
    }
    
    pub fn registerLayer(self: *CustomLayerRegistry, name: []const u8, layer_ptr: *anyopaque, forward_fn: *anyopaque, backward_fn: *anyopaque) !void {
        try self.layers.put(name, layer_ptr);
        try self.forward_fns.put(name, forward_fn);
        try self.backward_fns.put(name, backward_fn);
    }
    
    pub fn getLayer(self: *CustomLayerRegistry, name: []const u8) ?*anyopaque {
        return self.layers.get(name);
    }
    
    pub fn getForwardFn(self: *CustomLayerRegistry, name: []const u8) ?*anyopaque {
        return self.forward_fns.get(name);
    }
    
    pub fn getBackwardFn(self: *CustomLayerRegistry, name: []const u8) ?*anyopaque {
        return self.backward_fns.get(name);
    }
};

/// Model scaling strategies
pub const ModelScaling = struct {
    pub const ScalingConfig = struct {
        width_multiplier: f32,
        depth_multiplier: f32,
        resolution_multiplier: f32,
    };
    
    pub fn efficientNetScaling(phi: f32) ScalingConfig {
        const alpha = 1.2;
        const beta = 1.1;
        const gamma = 1.15;
        
        return .{
            .width_multiplier = std.math.pow(f32, alpha, phi),
            .depth_multiplier = std.math.pow(f32, beta, phi),
            .resolution_multiplier = std.math.pow(f32, gamma, phi),
        };
    }
    
    pub fn scaleModel(base_config: ModelConfig, scaling: ScalingConfig, allocator: std.mem.Allocator) !ModelConfig {
        var scaled_hidden_sizes = try allocator.alloc(usize, base_config.hidden_sizes.len);
        for (base_config.hidden_sizes, 0..) |size, i| {
            scaled_hidden_sizes[i] = @intFromFloat(@as(f32, @floatFromInt(size)) * scaling.width_multiplier);
        }
        
        const scaled_depth = @intFromFloat(@as(f32, @floatFromInt(base_config.hidden_sizes.len)) * scaling.depth_multiplier);
        
        return ModelConfig{
            .name = base_config.name,
            .input_size = base_config.input_size,
            .hidden_sizes = scaled_hidden_sizes[0..@min(scaled_depth, scaled_hidden_sizes.len)],
            .output_size = base_config.output_size,
            .activations = base_config.activations,
            .dropout_rates = base_config.dropout_rates,
            .use_batch_norm = base_config.use_batch_norm,
            .use_layer_norm = base_config.use_layer_norm,
        };
    }
};

/// Batch size scaling
pub const BatchSizeScaling = struct {
    pub fn optimalBatchSize(base_batch_size: usize, memory_limit_bytes: usize, model_size_bytes: usize) usize {
        const max_possible = memory_limit_bytes / (model_size_bytes * 4); // Account for gradients and optimizer state
        return @min(base_batch_size, max_possible);
    }
    
    pub fn adaptiveBatchSize(current_loss: f32, target_loss: f32, current_batch_size: usize) usize {
        if (current_loss > target_loss * 1.1) {
            return @max(1, current_batch_size / 2); // Decrease batch size if loss is too high
        } else if (current_loss < target_loss * 0.9) {
            return @min(1024, current_batch_size * 2); // Increase batch size if loss is good
        }
        return current_batch_size;
    }
};

/// Mixed precision training support
pub const MixedPrecision = struct {
    pub const Precision = enum {
        fp32,
        fp16,
        bfloat16,
    };
    
    pub fn convertTensor(tensor: *trix.DataObject, target_precision: Precision) !trix.DataObject {
        switch (target_precision) {
            .fp32 => return tensor.*, // Already fp32
            .fp16 => {
                // Simplified fp16 conversion - in practice would need proper half-precision support
                var fp16_tensor = try trix.DataObject.init(tensor.allocator, tensor.shape.?.items, .f32);
                for (tensor.values.items, 0..) |val, i| {
                    fp16_tensor.values.items[i] = @floatCast(val);
                }
                return fp16_tensor;
            },
            .bfloat16 => {
                // Simplified bfloat16 conversion
                var bf16_tensor = try trix.DataObject.init(tensor.allocator, tensor.shape.?.items, .f32);
                for (tensor.values.items, 0..) |val, i| {
                    bf16_tensor.values.items[i] = @floatCast(val);
                }
                return bf16_tensor;
            },
        }
    }
    
    pub fn shouldUseMixedPrecision(model_size: usize, batch_size: usize) bool {
        // Use mixed precision for large models or large batches
        return model_size > 1000000 or batch_size > 64;
    }
};
