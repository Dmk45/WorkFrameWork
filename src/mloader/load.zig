const std = @import("std");
const layers = @import("../layers.zig");
const trix = @import("../matrix.zig");

const MAGIC = "SIGMODEL";
const VERSION: u32 = 1;

/// Deserialize model structure from JSON
pub fn deserializeStructure(allocator: std.mem.Allocator, json_str: []const u8) !struct {
    layers: std.array_list.Managed(@import("save.zig").LayerConfig),
    param_paths: std.array_list.Managed([]const u8),
} {
    const save_mod = @import("save.zig");
    
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
    defer parsed.deinit();

    const obj = parsed.value.object;
    
    var layers_list = std.array_list.Managed(save_mod.LayerConfig).init(allocator);
    var paths_list = std.array_list.Managed([]const u8).init(allocator);

    // Parse layers
    if (obj.get("layers")) |layers_val| {
        for (layers_val.array.items) |layer_val| {
            const layer_obj = layer_val.object;
            
            var config = save_mod.LayerConfig{
                .layer_type = std.meta.stringToEnum(layers.LayerType, layer_obj.get("type").?.string) orelse return error.UnknownLayerType,
                .index = @intCast(layer_obj.get("index").?.integer),
            };

            if (layer_obj.get("input_size")) |val| config.input_size = @intCast(val.integer);
            if (layer_obj.get("output_size")) |val| config.output_size = @intCast(val.integer);
            if (layer_obj.get("activation")) |val| config.activation = try allocator.dupe(u8, val.string);
            if (layer_obj.get("in_channels")) |val| config.in_channels = @intCast(val.integer);
            if (layer_obj.get("out_channels")) |val| config.out_channels = @intCast(val.integer);
            if (layer_obj.get("kernel_size")) |val| config.kernel_size = @intCast(val.integer);
            if (layer_obj.get("kernel_h")) |val| config.kernel_h = @intCast(val.integer);
            if (layer_obj.get("kernel_w")) |val| config.kernel_w = @intCast(val.integer);
            if (layer_obj.get("kernel_d")) |val| config.kernel_d = @intCast(val.integer);
            if (layer_obj.get("stride")) |val| config.stride = @intCast(val.integer);
            if (layer_obj.get("padding")) |val| config.padding = @intCast(val.integer);
            if (layer_obj.get("hidden_size")) |val| config.hidden_size = @intCast(val.integer);
            if (layer_obj.get("num_layers")) |val| config.num_layers = @intCast(val.integer);

            try layers_list.append(config);
        }
    }

    // Parse param paths
    if (obj.get("param_paths")) |paths_val| {
        for (paths_val.array.items) |path_val| {
            try paths_list.append(try allocator.dupe(u8, path_val.string));
        }
    }

    return .{
        .layers = layers_list,
        .param_paths = paths_list,
    };
}

/// Reconstruct model from structure
pub fn reconstructModel(allocator: std.mem.Allocator, structure: anytype) !layers.NeuralNetwork {
    var nn = try layers.NeuralNetwork.init(allocator);

    for (structure.layers.items) |config| {
        switch (config.layer_type) {
            .linear => {
                const input_size = config.input_size orelse return error.MissingConfig;
                const output_size = config.output_size orelse return error.MissingConfig;
                const activation = config.activation orelse "none";
                
                const layer = try layers.LinearLayer.init(allocator, input_size, output_size, activation, null, null);
                try nn.add(layer);
            },
            .conv1d => {
                const in_channels = config.in_channels orelse return error.MissingConfig;
                const out_channels = config.out_channels orelse return error.MissingConfig;
                const kernel_size = config.kernel_size orelse return error.MissingConfig;
                const stride = config.stride orelse 1;
                const padding = config.padding orelse 0;
                
                const layer = try layers.Conv1DLayer.init(allocator, in_channels, out_channels, kernel_size, stride, padding, null, null);
                try nn.add(layer);
            },
            .conv2d => {
                const in_channels = config.in_channels orelse return error.MissingConfig;
                const out_channels = config.out_channels orelse return error.MissingConfig;
                const kernel_h = config.kernel_h orelse return error.MissingConfig;
                const kernel_w = config.kernel_w orelse return error.MissingConfig;
                const stride = config.stride orelse 1;
                const padding = config.padding orelse 0;
                
                const layer = try layers.Conv2DLayer.init(allocator, in_channels, out_channels, kernel_h, kernel_w, stride, padding, null, null);
                try nn.add(layer);
            },
            .conv3d => {
                const in_channels = config.in_channels orelse return error.MissingConfig;
                const out_channels = config.out_channels orelse return error.MissingConfig;
                const kernel_d = config.kernel_d orelse return error.MissingConfig;
                const kernel_h = config.kernel_h orelse return error.MissingConfig;
                const kernel_w = config.kernel_w orelse return error.MissingConfig;
                const stride = config.stride orelse 1;
                const padding = config.padding orelse 0;
                
                const layer = try layers.Conv3DLayer.init(allocator, in_channels, out_channels, kernel_d, kernel_h, kernel_w, stride, padding, null, null);
                try nn.add(layer);
            },
            .lstm => {
                const input_size = config.input_size orelse return error.MissingConfig;
                const hidden_size = config.hidden_size orelse return error.MissingConfig;
                
                const layer = try layers.LSTMCell.init(allocator, input_size, hidden_size, null, null);
                try nn.add(layer);
            },
            .gru => {
                const input_size = config.input_size orelse return error.MissingConfig;
                const hidden_size = config.hidden_size orelse return error.MissingConfig;
                
                const layer = try layers.GRUCell.init(allocator, input_size, hidden_size, null, null);
                try nn.add(layer);
            },
        }
    }

    return nn;
}

/// Load parameters from binary format
pub fn loadParams(allocator: std.mem.Allocator, nn: *layers.NeuralNetwork, param_paths: *const std.array_list.Managed([]const u8), file: std.fs.File) !void {
    // Build param map from reconstructed model
    var param_map = std.StringHashMap(*trix.DataObject).init(allocator);
    defer {
        // Free all allocated path strings
        var key_iter = param_map.keyIterator();
        while (key_iter.next()) |key| {
            allocator.free(key.*);
        }
        param_map.deinit();
    }

    for (nn.layers.items, 0..) |layer, i| {
        switch (layer.layer_type) {
            .linear => {
                const linear = @as(*layers.LinearLayer, @ptrCast(@alignCast(layer.type)));
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                try param_map.put(weights_path, &linear.weights);
                try param_map.put(bias_path, &linear.bias);
            },
            .conv1d => {
                const conv1d = @as(*layers.Conv1DLayer, @ptrCast(@alignCast(layer.type)));
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                try param_map.put(weights_path, &conv1d.weights);
                try param_map.put(bias_path, &conv1d.bias);
            },
            .conv2d => {
                const conv2d = @as(*layers.Conv2DLayer, @ptrCast(@alignCast(layer.type)));
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                try param_map.put(weights_path, &conv2d.weights);
                try param_map.put(bias_path, &conv2d.bias);
            },
            .conv3d => {
                const conv3d = @as(*layers.Conv3DLayer, @ptrCast(@alignCast(layer.type)));
                const weights_path = try std.fmt.allocPrint(allocator, "layers.{d}.weights", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                try param_map.put(weights_path, &conv3d.weights);
                try param_map.put(bias_path, &conv3d.bias);
            },
            .lstm => {
                const lstm = @as(*layers.LSTMCell, @ptrCast(@alignCast(layer.type)));
                const w_ih_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_ih", .{i});
                const w_hh_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_hh", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                try param_map.put(w_ih_path, &lstm.w_ih);
                try param_map.put(w_hh_path, &lstm.w_hh);
                try param_map.put(bias_path, &lstm.bias);
            },
            .gru => {
                const gru = @as(*layers.GRUCell, @ptrCast(@alignCast(layer.type)));
                const w_ih_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_ih", .{i});
                const w_hh_path = try std.fmt.allocPrint(allocator, "layers.{d}.w_hh", .{i});
                const bias_path = try std.fmt.allocPrint(allocator, "layers.{d}.bias", .{i});
                try param_map.put(w_ih_path, &gru.w_ih);
                try param_map.put(w_hh_path, &gru.w_hh);
                try param_map.put(bias_path, &gru.bias);
            },
        }
    }

    // Read and load each parameter
    for (param_paths.items) |path| {
        // Read path
        var path_len_buf: [4]u8 = undefined;
        _ = try file.readAll(&path_len_buf);
        const path_len = std.mem.readInt(u32, &path_len_buf, .little);
        const loaded_path = try allocator.alloc(u8, path_len);
        _ = try file.readAll(loaded_path);
        
        // Verify path matches expected
        if (!std.mem.eql(u8, path, loaded_path)) {
            allocator.free(loaded_path);
            return error.PathMismatch;
        }
        allocator.free(loaded_path);

        // Read shape
        var rank_buf: [4]u8 = undefined;
        _ = try file.readAll(&rank_buf);
        const rank = std.mem.readInt(u32, &rank_buf, .little);
        var shape = try allocator.alloc(usize, rank);
        defer allocator.free(shape);
        for (0..rank) |j| {
            var dim_buf: [4]u8 = undefined;
            _ = try file.readAll(&dim_buf);
            shape[j] = std.mem.readInt(u32, &dim_buf, .little);
        }

        // Read dtype
        var dtype_buf: [1]u8 = undefined;
        _ = try file.readAll(&dtype_buf);
        const dtype = @as(trix.DType, @enumFromInt(dtype_buf[0]));

        // Read value count
        var value_count_buf: [4]u8 = undefined;
        _ = try file.readAll(&value_count_buf);
        const value_count = std.mem.readInt(u32, &value_count_buf, .little);

        // Find target parameter
        const target = param_map.get(path) orelse return error.ParamNotFound;

        // Validate shape matches
        if (target.shape.?.items.len != rank) return error.ShapeMismatch;
        for (shape, 0..) |s, j| {
            if (target.shape.?.items[j] != s) return error.ShapeMismatch;
        }

        // Validate dtype matches
        if (target.dtype != dtype) return error.DtypeMismatch;

        // Validate value count matches
        if (target.values.items.len != value_count) return error.ValueCountMismatch;

        // Load values
        for (0..value_count) |j| {
            var val_buf: [4]u8 = undefined;
            _ = try file.readAll(&val_buf);
            target.values.items[j] = @bitCast(std.mem.readInt(u32, &val_buf, .little));
        }
    }
}

/// Load complete model from file
pub fn loadModel(allocator: std.mem.Allocator, path: []const u8) !layers.NeuralNetwork {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // Read structure JSON length
    var json_len_buf: [4]u8 = undefined;
    _ = try file.readAll(&json_len_buf);
    const json_len = std.mem.readInt(u32, &json_len_buf, .little);

    // Read structure JSON
    const json_str = try allocator.alloc(u8, json_len);
    defer allocator.free(json_str);
    _ = try file.readAll(json_str);

    // Deserialize structure
    const structure = try deserializeStructure(allocator, json_str);
    defer {
        for (structure.layers.items) |*config| {
            if (config.activation) |act| structure.layers.allocator.free(act);
        }
        structure.layers.deinit();
        for (structure.param_paths.items) |param_path| {
            structure.param_paths.allocator.free(param_path);
        }
        structure.param_paths.deinit();
    }

    // Reconstruct model
    var nn = try reconstructModel(allocator, structure);
    errdefer nn.deinit();

    // Read and verify binary header
    var magic_buf: [8]u8 = undefined;
    _ = try file.readAll(&magic_buf);
    if (!std.mem.eql(u8, &magic_buf, MAGIC)) return error.InvalidMagic;

    var version_buf: [4]u8 = undefined;
    _ = try file.readAll(&version_buf);
    const version = std.mem.readInt(u32, &version_buf, .little);
    if (version != VERSION) return error.UnsupportedVersion;

    var param_count_buf: [4]u8 = undefined;
    _ = try file.readAll(&param_count_buf);
    _ = std.mem.readInt(u32, &param_count_buf, .little);

    // Load parameters
    try loadParams(allocator, &nn, &structure.param_paths, file);

    return nn;
}

