const std = @import("std");

pub const DatasetVTable = struct {
    row_count: *const fn (ctx: *const anyopaque) usize,
    feature_count: *const fn (ctx: *const anyopaque) usize,
    fill_row_features: *const fn (ctx: *const anyopaque, row_index: usize, out: []f32) anyerror!void,
    row_label: *const fn (ctx: *const anyopaque, row_index: usize) anyerror!f32,
};

pub const Dataset = struct {
    ctx: *const anyopaque,
    vtable: *const DatasetVTable,

    pub fn rowCount(self: Dataset) usize {
        return self.vtable.row_count(self.ctx);
    }

    pub fn featureCount(self: Dataset) usize {
        return self.vtable.feature_count(self.ctx);
    }

    pub fn fillRowFeatures(self: Dataset, row_index: usize, out: []f32) !void {
        return self.vtable.fill_row_features(self.ctx, row_index, out);
    }

    pub fn rowLabel(self: Dataset, row_index: usize) !f32 {
        return self.vtable.row_label(self.ctx, row_index);
    }
};

pub const CsvDataset = struct {
    allocator: std.mem.Allocator,
    features: std.ArrayList(f32),
    labels: std.ArrayList(f32),
    feature_count: usize,
    row_count: usize,

    pub fn loadFromCsv(allocator: std.mem.Allocator, path: []const u8, has_header: bool) !CsvDataset {
        return loadDelimited(allocator, path, ',', has_header);
    }

    pub fn loadFromText(
        allocator: std.mem.Allocator,
        path: []const u8,
        delimiter: u8,
        has_header: bool,
    ) !CsvDataset {
        return loadDelimited(allocator, path, delimiter, has_header);
    }

    pub fn loadFromBinaryF32(allocator: std.mem.Allocator, path: []const u8) !CsvDataset {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const reader = file.reader();

        const row_count_u32 = try reader.readInt(u32, .little);
        const feature_count_u32 = try reader.readInt(u32, .little);
        const row_count = @as(usize, row_count_u32);
        const feature_count = @as(usize, feature_count_u32);
        if (row_count == 0 or feature_count == 0) return error.InvalidBinaryDataset;

        var features = try std.ArrayList(f32).init(allocator);
        errdefer features.deinit();
        try features.resize(row_count * feature_count);

        var labels = try std.ArrayList(f32).init(allocator);
        errdefer labels.deinit();
        try labels.resize(row_count);

        for (0..row_count) |r| {
            const base = r * feature_count;
            for (0..feature_count) |c| {
                features.items[base + c] = try reader.readFloat(f32, .little);
            }
            labels.items[r] = try reader.readFloat(f32, .little);
        }

        return .{
            .allocator = allocator,
            .features = features,
            .labels = labels,
            .feature_count = feature_count,
            .row_count = row_count,
        };
    }

    pub fn loadImageFile(
        allocator: std.mem.Allocator,
        path: []const u8,
        label: f32,
    ) !CsvDataset {
        if (std.mem.endsWith(u8, path, ".jpeg") or std.mem.endsWith(u8, path, ".jpg") or std.mem.endsWith(u8, path, ".png") or std.mem.endsWith(u8, path, ".webp")) {
            return error.UnsupportedImageFormat;
        }
        if (!std.mem.endsWith(u8, path, ".pgm")) return error.UnsupportedImageFormat;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const max_bytes = std.math.maxInt(usize);
        const bytes = try file.readToEndAlloc(allocator, max_bytes);
        defer allocator.free(bytes);

        var tok = std.mem.tokenizeAny(u8, bytes, " \r\n\t");
        const magic = tok.next() orelse return error.InvalidImageFile;
        if (!std.mem.eql(u8, magic, "P2")) return error.InvalidImageFile;
        const width_s = tok.next() orelse return error.InvalidImageFile;
        const height_s = tok.next() orelse return error.InvalidImageFile;
        const max_val_s = tok.next() orelse return error.InvalidImageFile;
        const width = try std.fmt.parseInt(usize, width_s, 10);
        const height = try std.fmt.parseInt(usize, height_s, 10);
        const max_val = try std.fmt.parseInt(usize, max_val_s, 10);
        if (max_val == 0) return error.InvalidImageFile;

        const feature_count = width * height;
        var features = try std.ArrayList(f32).init(allocator);
        errdefer features.deinit();
        try features.resize(feature_count);

        for (0..feature_count) |i| {
            const px_s = tok.next() orelse return error.InvalidImageFile;
            const px = try std.fmt.parseInt(usize, px_s, 10);
            features.items[i] = @as(f32, @floatFromInt(px)) / @as(f32, @floatFromInt(max_val));
        }

        var labels = try std.ArrayList(f32).init(allocator);
        errdefer labels.deinit();
        try labels.append(label);

        return .{
            .allocator = allocator,
            .features = features,
            .labels = labels,
            .feature_count = feature_count,
            .row_count = 1,
        };
    }

    fn loadDelimited(
        allocator: std.mem.Allocator,
        path: []const u8,
        delimiter: u8,
        has_header: bool,
    ) !CsvDataset {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const max_bytes = std.math.maxInt(usize);
        const file_content = try file.readToEndAlloc(allocator, max_bytes);
        defer allocator.free(file_content);

        var features = std.ArrayList(f32).init(allocator);
        errdefer features.deinit();
        var labels = std.ArrayList(f32).init(allocator);
        errdefer labels.deinit();

        var maybe_feature_count: ?usize = null;
        var row_count: usize = 0;
        var line_index: usize = 0;

        var lines = std.mem.splitScalar(u8, file_content, '\n');
        while (lines.next()) |raw_line| : (line_index += 1) {
            const line = std.mem.trim(u8, raw_line, " \r\t");
            if (line.len == 0) continue;
            if (has_header and line_index == 0) continue;

            var values = std.ArrayList(f32).init(allocator);
            defer values.deinit();
            var columns = std.mem.splitScalar(u8, line, delimiter);
            while (columns.next()) |col| {
                const token = std.mem.trim(u8, col, " \t\r");
                if (token.len == 0) return error.InvalidCsvValue;
                try values.append(std.fmt.parseFloat(f32, token) catch return error.InvalidCsvValue);
            }

            if (values.items.len < 2) return error.InvalidCsvRow;
            const current_feature_count = values.items.len - 1;
            if (maybe_feature_count) |known| {
                if (known != current_feature_count) return error.InconsistentFeatureCount;
            } else maybe_feature_count = current_feature_count;

            try features.appendSlice(values.items[0..current_feature_count]);
            try labels.append(values.items[current_feature_count]);
            row_count += 1;
        }

        if (row_count == 0) return error.EmptyDataset;
        return .{
            .allocator = allocator,
            .features = features,
            .labels = labels,
            .feature_count = maybe_feature_count.?,
            .row_count = row_count,
        };
    }

    pub fn asDataset(self: *const CsvDataset) Dataset {
        return .{ .ctx = self, .vtable = &vtable };
    }

    pub fn deinit(self: *CsvDataset) void {
        self.features.deinit();
        self.labels.deinit();
    }

    pub fn rowFeatures(self: *const CsvDataset, row_index: usize) ![]const f32 {
        if (row_index >= self.row_count) return error.IndexOutOfBounds;
        const start = row_index * self.feature_count;
        return self.features.items[start .. start + self.feature_count];
    }

    pub fn rowLabel(self: *const CsvDataset, row_index: usize) !f32 {
        if (row_index >= self.row_count) return error.IndexOutOfBounds;
        return self.labels.items[row_index];
    }

    const vtable = DatasetVTable{
        .row_count = datasetRowCount,
        .feature_count = datasetFeatureCount,
        .fill_row_features = datasetFillRowFeatures,
        .row_label = datasetRowLabel,
    };

    fn datasetRowCount(ctx: *const anyopaque) usize {
        const self: *const CsvDataset = @ptrCast(@alignCast(ctx));
        return self.row_count;
    }
    fn datasetFeatureCount(ctx: *const anyopaque) usize {
        const self: *const CsvDataset = @ptrCast(@alignCast(ctx));
        return self.feature_count;
    }
    fn datasetFillRowFeatures(ctx: *const anyopaque, row_index: usize, out: []f32) !void {
        const self: *const CsvDataset = @ptrCast(@alignCast(ctx));
        if (out.len != self.feature_count) return error.InvalidFeatureBuffer;
        const row = try self.rowFeatures(row_index);
        @memcpy(out, row);
    }
    fn datasetRowLabel(ctx: *const anyopaque, row_index: usize) !f32 {
        const self: *const CsvDataset = @ptrCast(@alignCast(ctx));
        return self.rowLabel(row_index);
    }
};

pub const StandardScaler = struct {
    allocator: std.mem.Allocator,
    means: std.ArrayList(f32),
    stds: std.ArrayList(f32),

    pub fn fit(allocator: std.mem.Allocator, dataset: Dataset, indices: []const usize) !StandardScaler {
        if (indices.len == 0) return error.EmptyDataset;
        const feature_count = dataset.featureCount();
        var means = std.ArrayList(f32).init(allocator);
        errdefer means.deinit();
        try means.resize(feature_count);
        @memset(means.items, 0.0);

        const row_buf = try allocator.alloc(f32, feature_count);
        defer allocator.free(row_buf);
        for (indices) |idx| {
            try dataset.fillRowFeatures(idx, row_buf);
            for (0..feature_count) |f| means.items[f] += row_buf[f];
        }
        const n = @as(f32, @floatFromInt(indices.len));
        for (0..feature_count) |f| means.items[f] /= n;

        var stds = std.ArrayList(f32).init(allocator);
        errdefer stds.deinit();
        try stds.resize(feature_count);
        @memset(stds.items, 0.0);
        for (indices) |idx| {
            try dataset.fillRowFeatures(idx, row_buf);
            for (0..feature_count) |f| {
                const d = row_buf[f] - means.items[f];
                stds.items[f] += d * d;
            }
        }
        for (0..feature_count) |f| {
            const variance = stds.items[f] / n;
            stds.items[f] = @max(1e-8, @sqrt(variance));
        }
        return .{ .allocator = allocator, .means = means, .stds = stds };
    }

    pub fn transform(self: *const StandardScaler, values: []f32) void {
        for (0..values.len) |i| values[i] = (values[i] - self.means.items[i]) / self.stds.items[i];
    }

    pub fn deinit(self: *StandardScaler) void {
        self.means.deinit();
        self.stds.deinit();
    }
};

pub const MinMaxScaler = struct {
    allocator: std.mem.Allocator,
    mins: std.ArrayList(f32),
    maxs: std.ArrayList(f32),

    pub fn fit(allocator: std.mem.Allocator, dataset: Dataset, indices: []const usize) !MinMaxScaler {
        if (indices.len == 0) return error.EmptyDataset;
        const feature_count = dataset.featureCount();
        var mins = std.ArrayList(f32).init(allocator);
        errdefer mins.deinit();
        try mins.resize(feature_count);
        var maxs = std.ArrayList(f32).init(allocator);
        errdefer maxs.deinit();
        try maxs.resize(feature_count);
        @memset(mins.items, std.math.inf(f32));
        @memset(maxs.items, -std.math.inf(f32));

        const row_buf = try allocator.alloc(f32, feature_count);
        defer allocator.free(row_buf);
        for (indices) |idx| {
            try dataset.fillRowFeatures(idx, row_buf);
            for (0..feature_count) |f| {
                mins.items[f] = @min(mins.items[f], row_buf[f]);
                maxs.items[f] = @max(maxs.items[f], row_buf[f]);
            }
        }
        return .{ .allocator = allocator, .mins = mins, .maxs = maxs };
    }

    pub fn transform(self: *const MinMaxScaler, values: []f32) void {
        for (0..values.len) |i| {
            const range = @max(1e-8, self.maxs.items[i] - self.mins.items[i]);
            values[i] = (values[i] - self.mins.items[i]) / range;
        }
    }

    pub fn deinit(self: *MinMaxScaler) void {
        self.mins.deinit();
        self.maxs.deinit();
    }
};

pub const Augmentation = union(enum) {
    none,
    gaussian_noise: f32,
    random_feature_dropout: f32,

    pub fn apply(self: Augmentation, values: []f32, rng: *std.Random) void {
        switch (self) {
            .none => {},
            .gaussian_noise => |stddev| {
                for (0..values.len) |i| {
                    values[i] += rng.floatNorm(f32) * stddev;
                }
            },
            .random_feature_dropout => |p| {
                for (0..values.len) |i| {
                    if (rng.float(f32) < p) values[i] = 0.0;
                }
            },
        }
    }
};

pub const DataSplit = struct {
    allocator: std.mem.Allocator,
    train_indices: std.ArrayList(usize),
    val_indices: std.ArrayList(usize),
    test_indices: std.ArrayList(usize),

    pub fn deinit(self: *DataSplit) void {
        self.train_indices.deinit();
        self.val_indices.deinit();
        self.test_indices.deinit();
    }
};

pub const DistributedShard = struct {
    world_size: usize = 1,
    rank: usize = 0,
};

pub fn splitIndices(
    allocator: std.mem.Allocator,
    total_rows: usize,
    train_ratio: f32,
    val_ratio: f32,
    seed: u64,
) !DataSplit {
    if (train_ratio <= 0.0 or val_ratio < 0.0) return error.InvalidSplitRatio;
    if (train_ratio + val_ratio > 1.0) return error.InvalidSplitRatio;

    var shuffled = std.ArrayList(usize).init(allocator);
    defer shuffled.deinit();
    try shuffled.ensureTotalCapacity(total_rows);
    for (0..total_rows) |idx| {
        shuffled.appendAssumeCapacity(idx);
    }

    var prng = std.Random.DefaultPrng.init(seed);
    var rng = prng.random();
    fisherYates(usize, shuffled.items, &rng);

    const train_count = @as(usize, @intFromFloat(@as(f64, @floatFromInt(total_rows)) * @as(f64, train_ratio)));
    const val_count = @as(usize, @intFromFloat(@as(f64, @floatFromInt(total_rows)) * @as(f64, val_ratio)));
    const safe_train = @min(train_count, total_rows);
    const safe_val = @min(val_count, total_rows - safe_train);
    const test_start = safe_train + safe_val;

    var train_indices = std.ArrayList(usize).init(allocator);
    errdefer train_indices.deinit();
    try train_indices.appendSlice(shuffled.items[0..safe_train]);

    var val_indices = std.ArrayList(usize).init(allocator);
    errdefer val_indices.deinit();
    try val_indices.appendSlice(shuffled.items[safe_train..test_start]);

    var test_indices = std.ArrayList(usize).init(allocator);
    errdefer test_indices.deinit();
    try test_indices.appendSlice(shuffled.items[test_start..]);

    return .{
        .allocator = allocator,
        .train_indices = train_indices,
        .val_indices = val_indices,
        .test_indices = test_indices,
    };
}

pub fn shardIndices(
    allocator: std.mem.Allocator,
    indices: []const usize,
    shard: DistributedShard,
) !std.ArrayList(usize) {
    if (shard.world_size == 0 or shard.rank >= shard.world_size) return error.InvalidDistributedShard;
    var out = std.ArrayList(usize).init(allocator);
    errdefer out.deinit();
    for (indices, 0..) |idx, pos| {
        if (pos % shard.world_size == shard.rank) try out.append(idx);
    }
    return out;
}

pub const BatchIterator = struct {
    indices: []const usize,
    batch_size: usize,
    position: usize = 0,

    pub fn init(indices: []const usize, batch_size: usize) !BatchIterator {
        if (batch_size == 0) return error.InvalidBatchSize;
        return .{
            .indices = indices,
            .batch_size = batch_size,
            .position = 0,
        };
    }

    pub fn reset(self: *BatchIterator) void {
        self.position = 0;
    }

    pub fn next(self: *BatchIterator) ?[]const usize {
        if (self.position >= self.indices.len) return null;
        const end = @min(self.position + self.batch_size, self.indices.len);
        const batch = self.indices[self.position..end];
        self.position = end;
        return batch;
    }
};

pub const ShuffleStrategy = enum {
    in_order,
    random,
    stratified,
};

pub const DataLoaderConfig = struct {
    batch_size: usize,
    strategy: ShuffleStrategy = .in_order,
    seed: u64 = 0,
    prefetch: bool = false,
    use_cache: bool = false,
    shard: DistributedShard = .{},
    augmentation: Augmentation = .none,
};

pub const Batch = struct {
    features: []const f32,
    labels: []const f32,
    batch_len: usize,
    feature_count: usize,
};

pub const DataLoader = struct {
    allocator: std.mem.Allocator,
    dataset: Dataset,
    config: DataLoaderConfig,
    epoch_indices: std.ArrayList(usize),
    position: usize,
    rng: std.Random.DefaultPrng,
    row_buffer: []f32,
    batch_features: std.ArrayList(f32),
    batch_labels: std.ArrayList(f32),
    prefetched_features: std.ArrayList(f32),
    prefetched_labels: std.ArrayList(f32),
    prefetched_len: usize,
    has_prefetched: bool,
    cache: std.AutoHashMap(usize, std.ArrayList(f32)),

    pub fn init(
        allocator: std.mem.Allocator,
        dataset: Dataset,
        base_indices: []const usize,
        config: DataLoaderConfig,
    ) !DataLoader {
        if (config.batch_size == 0) return error.InvalidBatchSize;
        var sharded = try shardIndices(allocator, base_indices, config.shard);
        errdefer sharded.deinit();

        var dl = DataLoader{
            .allocator = allocator,
            .dataset = dataset,
            .config = config,
            .epoch_indices = sharded,
            .position = 0,
            .rng = std.Random.DefaultPrng.init(config.seed),
            .row_buffer = try allocator.alloc(f32, dataset.featureCount()),
            .batch_features = std.ArrayList(f32).init(allocator),
            .batch_labels = std.ArrayList(f32).init(allocator),
            .prefetched_features = std.ArrayList(f32).init(allocator),
            .prefetched_labels = std.ArrayList(f32).init(allocator),
            .prefetched_len = 0,
            .has_prefetched = false,
            .cache = std.AutoHashMap(usize, std.ArrayList(f32)).init(allocator),
        };
        try dl.reset();
        return dl;
    }

    pub fn deinit(self: *DataLoader) void {
        var it = self.cache.valueIterator();
        while (it.next()) |arr| arr.deinit();
        self.cache.deinit();
        self.prefetched_features.deinit();
        self.prefetched_labels.deinit();
        self.batch_features.deinit();
        self.batch_labels.deinit();
        self.epoch_indices.deinit();
        self.allocator.free(self.row_buffer);
    }

    pub fn reset(self: *DataLoader) !void {
        self.position = 0;
        self.has_prefetched = false;
        try self.applyStrategy();
        if (self.config.prefetch) try self.prefetchNext();
    }

    pub fn next(self: *DataLoader) !?Batch {
        if (self.config.prefetch) {
            if (!self.has_prefetched) return null;
            self.batch_features.clearRetainingCapacity();
            self.batch_labels.clearRetainingCapacity();
            try self.batch_features.appendSlice(self.prefetched_features.items);
            try self.batch_labels.appendSlice(self.prefetched_labels.items);
            const out_len = self.prefetched_len;
            self.has_prefetched = false;
            try self.prefetchNext();
            return .{
                .features = self.batch_features.items,
                .labels = self.batch_labels.items,
                .batch_len = out_len,
                .feature_count = self.dataset.featureCount(),
            };
        }
        return self.computeNextBatch();
    }

    fn prefetchNext(self: *DataLoader) !void {
        const maybe = try self.computeNextBatchInto(&self.prefetched_features, &self.prefetched_labels);
        if (maybe) |len| {
            self.prefetched_len = len;
            self.has_prefetched = true;
        } else self.has_prefetched = false;
    }

    fn computeNextBatch(self: *DataLoader) !?Batch {
        const maybe = try self.computeNextBatchInto(&self.batch_features, &self.batch_labels);
        if (maybe) |batch_len| {
            return .{
                .features = self.batch_features.items,
                .labels = self.batch_labels.items,
                .batch_len = batch_len,
                .feature_count = self.dataset.featureCount(),
            };
        }
        return null;
    }

    fn computeNextBatchInto(
        self: *DataLoader,
        features_buf: *std.ArrayList(f32),
        labels_buf: *std.ArrayList(f32),
    ) !?usize {
        if (self.position >= self.epoch_indices.items.len) return null;
        features_buf.clearRetainingCapacity();
        labels_buf.clearRetainingCapacity();
        const start = self.position;
        const end = @min(self.position + self.config.batch_size, self.epoch_indices.items.len);
        var rng = self.rng.random();
        var r = self.position;
        while (r < end) : (r += 1) {
            const idx = self.epoch_indices.items[r];
            try self.getRow(idx, self.row_buffer);
            self.config.augmentation.apply(self.row_buffer, &rng);
            try features_buf.appendSlice(self.row_buffer);
            try labels_buf.append(try self.dataset.rowLabel(idx));
        }
        self.position = end;
        return end - start;
    }

    fn getRow(self: *DataLoader, idx: usize, out: []f32) !void {
        if (self.config.use_cache) {
            if (self.cache.getPtr(idx)) |cached| {
                @memcpy(out, cached.items);
                return;
            }
        }
        try self.dataset.fillRowFeatures(idx, out);
        if (self.config.use_cache) {
            var arr = std.ArrayList(f32).init(self.allocator);
            try arr.appendSlice(out);
            try self.cache.put(idx, arr);
        }
    }

    fn applyStrategy(self: *DataLoader) !void {
        switch (self.config.strategy) {
            .in_order => {},
            .random => {
                var rng = self.rng.random();
                fisherYates(usize, self.epoch_indices.items, &rng);
            },
            .stratified => try self.stratifiedOrder(),
        }
    }

    fn stratifiedOrder(self: *DataLoader) !void {
        var groups = std.AutoHashMap(u32, std.ArrayList(usize)).init(self.allocator);
        defer {
            var g_it = groups.valueIterator();
            while (g_it.next()) |arr| arr.deinit();
            groups.deinit();
        }
        for (self.epoch_indices.items) |idx| {
            const label = try self.dataset.rowLabel(idx);
            const key: u32 = @bitCast(label);
            if (groups.getPtr(key)) |arr| {
                try arr.append(idx);
            } else {
                var arr = std.ArrayList(usize).init(self.allocator);
                defer arr.deinit();
                try arr.append(idx);
                try groups.put(key, arr);
            }
        }

        var keys = std.ArrayList(u32).init(self.allocator);
        defer keys.deinit();
        var it = groups.iterator();
        while (it.next()) |entry| try keys.append(entry.key_ptr.*);

        var rng = self.rng.random();
        for (keys.items) |key| if (groups.getPtr(key)) |arr| fisherYates(usize, arr.items, &rng);

        const total = self.epoch_indices.items.len;
        self.epoch_indices.clearRetainingCapacity();
        var cursor = std.AutoHashMap(u32, usize).init(self.allocator);
        defer cursor.deinit();
        for (keys.items) |key| try cursor.put(key, 0);

        var added: usize = 0;
        while (added < total) {
            var any = false;
            for (keys.items) |key| {
                var pos = cursor.get(key).?;
                const arr = groups.getPtr(key).?;
                if (pos < arr.items.len) {
                    try self.epoch_indices.append(arr.items[pos]);
                    pos += 1;
                    try cursor.put(key, pos);
                    added += 1;
                    any = true;
                }
            }
            if (!any) break;
        }
    }
};

fn fisherYates(comptime T: type, values: []T, rng: *std.Random) void {
    if (values.len <= 1) return;

    var i: usize = values.len - 1;
    while (true) {
        const j = rng.uintLessThan(usize, i + 1);
        std.mem.swap(T, &values[i], &values[j]);
        if (i == 0) break;
        i -= 1;
    }
}
