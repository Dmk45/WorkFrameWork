const std = @import("std");
const layers = @import("../layers.zig");
const trix = @import("../matrix.zig");

const MAGIC = "SIGMODEL";
const VERSION: u32 = 1;

/// Layer configuration for serialization
pub const LayerConfig = struct {
    layer_type: layers.LayerType,
    index: usize,
    // Linear layer config
    input_size: ?usize = null,
    output_size: ?usize = null,
    activation: ?[]const u8 = null,
    // Conv layer config
    in_channels: ?usize = null,
    out_channels: ?usize = null,
    kernel_size: ?usize = null,
    kernel_h: ?usize = null,
    kernel_w: ?usize = null,
    kernel_d: ?usize = null,
    stride: ?usize = null,
    padding: ?usize = null,
    // RNN layer config
    hidden_size: ?usize = null,
    num_layers: ?usize = null,
};

/// Model structure for JSON serialization
pub const ModelStructure = struct {
    layers: std.array_list.Managed(LayerConfig),
    param_paths: std.array_list.Managed([]const u8),

    pub fn init(allocator: std.mem.Allocator) ModelStructure {
        return .{
            .layers = std.array_list.Managed(LayerConfig).init(allocator),
            .param_paths = std.array_list.Managed([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ModelStructure) void {
        for (self.layers.items) |*config| {
            if (config.activation) |act| self.layers.allocator.free(act);
        }
        self.layers.deinit();
        for (self.param_paths.items) |path| {
            self.param_paths.allocator.free(path);
        }
        self.param_paths.deinit();
    }
};

/// Extract parameters from a NeuralNetwork and assign paths
pub fn extractParams(allocator: std.mem.Allocator, nn: *layers.NeuralNetwork) !ModelStructure {
    var structure = ModelStructure.init(allocator);

    for (nn.layers.items, 0..) |layer, i| {
        const layer_type = layer.layer_type;

        switch (layer_type) {
            .linear => {
                const linear = @as(*layers.LinearLayer, @ptrCast(@alignCast(layer.type)));
                
                // Set param paths
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                linear.weights.param_path = weights_path;
                linear.bias.param_path = bias_path;

                try structure.param_paths.append(weights_path);
                try structure.param_paths.append(bias_path);

                // Add layer config
                try structure.layers.append(.{
                    .layer_type = .linear,
                    .index = i,
                    .input_size = linear.config.input_size,
                    .output_size = linear.config.output_size,
                    .activation = try allocator.dupe(u8, linear.config.activation),
                });
            },
            .conv1d => {
                const conv1d = @as(*layers.Conv1DLayer, @ptrCast(@alignCast(layer.type)));
                
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                conv1d.weights.param_path = weights_path;
                conv1d.bias.param_path = bias_path;

                try structure.param_paths.append(weights_path);
                try structure.param_paths.append(bias_path);

                try structure.layers.append(.{
                    .layer_type = .conv1d,
                    .index = i,
                    .in_channels = conv1d.in_channels,
                    .out_channels = conv1d.out_channels,
                    .kernel_size = conv1d.kernel_size,
                    .stride = conv1d.stride,
                    .padding = conv1d.padding,
                });
            },
            .conv2d => {
                const conv2d = @as(*layers.Conv2DLayer, @ptrCast(@alignCast(layer.type)));
                
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                conv2d.weights.param_path = weights_path;
                conv2d.bias.param_path = bias_path;

                try structure.param_paths.append(weights_path);
                try structure.param_paths.append(bias_path);

                try structure.layers.append(.{
                    .layer_type = .conv2d,
                    .index = i,
                    .in_channels = conv2d.in_channels,
                    .out_channels = conv2d.out_channels,
                    .kernel_h = conv2d.kernel_h,
                    .kernel_w = conv2d.kernel_w,
                    .stride = conv2d.stride,
                    .padding = conv2d.padding,
                });
            },
            .conv3d => {
                const conv3d = @as(*layers.Conv3DLayer, @ptrCast(@alignCast(layer.type)));
                
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                conv3d.weights.param_path = weights_path;
                conv3d.bias.param_path = bias_path;

                try structure.param_paths.append(weights_path);
                try structure.param_paths.append(bias_path);

                try structure.layers.append(.{
                    .layer_type = .conv3d,
                    .index = i,
                    .in_channels = conv3d.in_channels,
                    .out_channels = conv3d.out_channels,
                    .kernel_d = conv3d.kernel_d,
                    .kernel_h = conv3d.kernel_h,
                    .kernel_w = conv3d.kernel_w,
                    .stride = conv3d.stride,
                    .padding = conv3d.padding,
                });
            },
            .lstm => {
                // Check if it's LSTMCell or LSTMLayer
                // For simplicity, treat as LSTMCell for now
                const lstm = @as(*layers.LSTMCell, @ptrCast(@alignCast(layer.type)));
                
                const w_ih_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_ih", .{i});
                const w_hh_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_hh", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                lstm.w_ih.param_path = w_ih_path;
                lstm.w_hh.param_path = w_hh_path;
                lstm.bias.param_path = bias_path;

                try structure.param_paths.append(w_ih_path);
                try structure.param_paths.append(w_hh_path);
                try structure.param_paths.append(bias_path);

                try structure.layers.append(.{
                    .layer_type = .lstm,
                    .index = i,
                    .input_size = lstm.input_size,
                    .hidden_size = lstm.hidden_size,
                });
            },
            .gru => {
                const gru = @as(*layers.GRUCell, @ptrCast(@alignCast(layer.type)));
                
                const w_ih_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_ih", .{i});
                const w_hh_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_hh", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                gru.w_ih.param_path = w_ih_path;
                gru.w_hh.param_path = w_hh_path;
                gru.bias.param_path = bias_path;

                try structure.param_paths.append(w_ih_path);
                try structure.param_paths.append(w_hh_path);
                try structure.param_paths.append(bias_path);

                try structure.layers.append(.{
                    .layer_type = .gru,
                    .index = i,
                    .input_size = gru.input_size,
                    .hidden_size = gru.hidden_size,
                });
            },
        }
    }

    return structure;
}

/// Serialize model structure to JSON
pub fn serializeStructure(allocator: std.mem.Allocator, structure: *ModelStructure) ![]const u8 {
    var array_list = std.array_list.Managed(u8).init(allocator);
    const writer = array_list.writer();

    try writer.writeAll("{\n");
    try writer.writeAll("  \"layers\": [\n");

    for (structure.layers.items, 0..) |config, i| {
        if (i > 0) try writer.writeAll(",\n");
        try writer.writeAll("    {\n");
        try writer.print("      \"type\": \"{s}\",\n", .{@tagName(config.layer_type)});
        try writer.print("      \"index\": {d}", .{config.index});

        if (config.input_size) |size| try writer.print(",\n      \"input_size\": {d}", .{size});
        if (config.output_size) |size| try writer.print(",\n      \"output_size\": {d}", .{size});
        if (config.activation) |act| try writer.print(",\n      \"activation\": \"{s}\"", .{act});
        if (config.in_channels) |ch| try writer.print(",\n      \"in_channels\": {d}", .{ch});
        if (config.out_channels) |ch| try writer.print(",\n      \"out_channels\": {d}", .{ch});
        if (config.kernel_size) |ks| try writer.print(",\n      \"kernel_size\": {d}", .{ks});
        if (config.kernel_h) |kh| try writer.print(",\n      \"kernel_h\": {d}", .{kh});
        if (config.kernel_w) |kw| try writer.print(",\n      \"kernel_w\": {d}", .{kw});
        if (config.kernel_d) |kd| try writer.print(",\n      \"kernel_d\": {d}", .{kd});
        if (config.stride) |s| try writer.print(",\n      \"stride\": {d}", .{s});
        if (config.padding) |p| try writer.print(",\n      \"padding\": {d}", .{p});
        if (config.hidden_size) |hs| try writer.print(",\n      \"hidden_size\": {d}", .{hs});
        if (config.num_layers) |nl| try writer.print(",\n      \"num_layers\": {d}", .{nl});

        try writer.writeAll("\n    }");
    }

    try writer.writeAll("\n  ],\n");
    try writer.writeAll("  \"param_paths\": [\n");

    for (structure.param_paths.items, 0..) |path, i| {
        if (i > 0) try writer.writeAll(",\n");
        try writer.print("    \"{s}\"", .{path});
    }

    try writer.writeAll("\n  ]\n");
    try writer.writeAll("}");

    return array_list.toOwnedSlice();
}

/// Serialize parameters to binary format
pub fn serializeParams(allocator: std.mem.Allocator, nn: *layers.NeuralNetwork, structure: *ModelStructure, file: std.fs.File) !void {
    // Write header
    try file.writeAll(MAGIC);
    
    var version_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &version_buf, VERSION, .little);
    try file.writeAll(&version_buf);
    
    var count_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &count_buf, @intCast(structure.param_paths.items.len), .little);
    try file.writeAll(&count_buf);

    // Collect all parameters with their paths
    var param_map = std.StringHashMap(*trix.DataObject).init(allocator);
    defer param_map.deinit();

    for (nn.layers.items) |layer| {
        switch (layer.layer_type) {
            .linear => {
                const linear = @as(*layers.LinearLayer, @ptrCast(@alignCast(layer.type)));
                if (linear.weights.param_path) |path| try param_map.put(path, &linear.weights);
                if (linear.bias.param_path) |path| try param_map.put(path, &linear.bias);
            },
            .conv1d => {
                const conv1d = @as(*layers.Conv1DLayer, @ptrCast(@alignCast(layer.type)));
                if (conv1d.weights.param_path) |path| try param_map.put(path, &conv1d.weights);
                if (conv1d.bias.param_path) |path| try param_map.put(path, &conv1d.bias);
            },
            .conv2d => {
                const conv2d = @as(*layers.Conv2DLayer, @ptrCast(@alignCast(layer.type)));
                if (conv2d.weights.param_path) |path| try param_map.put(path, &conv2d.weights);
                if (conv2d.bias.param_path) |path| try param_map.put(path, &conv2d.bias);
            },
            .conv3d => {
                const conv3d = @as(*layers.Conv3DLayer, @ptrCast(@alignCast(layer.type)));
                if (conv3d.weights.param_path) |path| try param_map.put(path, &conv3d.weights);
                if (conv3d.bias.param_path) |path| try param_map.put(path, &conv3d.bias);
            },
            .lstm => {
                const lstm = @as(*layers.LSTMCell, @ptrCast(@alignCast(layer.type)));
                if (lstm.w_ih.param_path) |path| try param_map.put(path, &lstm.w_ih);
                if (lstm.w_hh.param_path) |path| try param_map.put(path, &lstm.w_hh);
                if (lstm.bias.param_path) |path| try param_map.put(path, &lstm.bias);
            },
            .gru => {
                const gru = @as(*layers.GRUCell, @ptrCast(@alignCast(layer.type)));
                if (gru.w_ih.param_path) |path| try param_map.put(path, &gru.w_ih);
                if (gru.w_hh.param_path) |path| try param_map.put(path, &gru.w_hh);
                if (gru.bias.param_path) |path| try param_map.put(path, &gru.bias);
            },
        }
    }

    // Write each parameter
    for (structure.param_paths.items) |path| {
        const param = param_map.get(path) orelse return error.ParamNotFound;

        // Write path
        var path_len_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &path_len_buf, @intCast(path.len), .little);
        try file.writeAll(&path_len_buf);
        try file.writeAll(path);

        // Write shape
        const shape = param.shape.?.items;
        var rank_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &rank_buf, @intCast(shape.len), .little);
        try file.writeAll(&rank_buf);
        for (shape) |dim| {
            var dim_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &dim_buf, @intCast(dim), .little);
            try file.writeAll(&dim_buf);
        }

        // Write dtype
        try file.writeAll(&[_]u8{@intFromEnum(param.dtype)});

        // Write values
        var value_count_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &value_count_buf, @intCast(param.values.items.len), .little);
        try file.writeAll(&value_count_buf);
        for (param.values.items) |val| {
            try file.writeAll(std.mem.asBytes(&val));
        }
    }
}

/// Save complete model to file (structure + params)
pub fn saveModel(allocator: std.mem.Allocator, nn: *layers.NeuralNetwork, path: []const u8) !void {
    // Extract parameters and structure
    var structure = try extractParams(allocator, nn);
    defer structure.deinit();

    // Serialize structure to JSON
    const structure_json = try serializeStructure(allocator, &structure);
    defer allocator.free(structure_json);

    // Open file for writing
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    // Write structure JSON length and JSON
    var len_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_buf, @intCast(structure_json.len), .little);
    try file.writeAll(&len_buf);
    try file.writeAll(structure_json);

    // Write parameters in binary format
    try serializeParams(allocator, nn, &structure, file);
}
