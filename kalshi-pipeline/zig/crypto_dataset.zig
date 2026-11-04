const std = @import("std");
const modelwork2 = @import("modelwork2");
const trix = modelwork2.matrix;

pub const BtcPriceRecord = struct {
    timestamp: []const u8,
    price: f32,
};

pub const BtcPriceDataset = struct {
    crypto: []const u8,
    quote_currency: []const u8,
    product_id: []const u8,
    requested_days: u32,
    window_start: []const u8,
    window_end: []const u8,
    granularity: ?[]const u8 = null,
    price_source: []const u8,
    aligned_with: ?[]const u8 = null,
    source: []const u8,
    target_definition: []const u8,
    row_count: usize,
    y_rows: []BtcPriceRecord,
};

pub const KalshiTrainingRow = struct {
    ticker: []const u8,
    timestamp: []const u8,
    probability: f32,
    lookahead_seconds: i32,
    target_btc_price: ?f32,
};

pub const KalshiDataset = struct {
    crypto: []const u8,
    requested_days: u32,
    window_start: []const u8,
    window_end: []const u8,
    ticker_chunks: []TickerChunk,
    training_rows: []KalshiTrainingRow,
};

pub const TickerChunk = struct {
    ticker: []const u8,
    lookahead_seconds: i32,
    window_start: ?[]const u8,
    window_end: ?[]const u8,
    probability_records: []ProbabilityRecord,
};

pub const ProbabilityRecord = struct {
    ticker: []const u8,
    timestamp: []const u8,
    probability: f32,
    lookahead_seconds: i32,
};

pub const FeatureConfig = struct {
    probability: bool = true,
    lookahead_seconds: bool = true,

    pub fn countFeatures(self: FeatureConfig) usize {
        var count: usize = 0;
        if (self.probability) count += 1;
        if (self.lookahead_seconds) count += 1;
        return count;
    }
};

pub const AlignedDataset = struct {
    allocator: std.mem.Allocator,
    x_tensor: trix.DataObject,
    y_tensor: trix.DataObject,

    pub fn deinit(self: *AlignedDataset) void {
        self.x_tensor.deinit();
        self.y_tensor.deinit();
    }
};

pub fn isLeapYear(year: i32) bool {
    return @mod(year, 4) == 0 and (@mod(year, 100) != 0 or @mod(year, 400) == 0);
}

pub fn parseIso8601ToSeconds(str: []const u8) !f64 {
    // Expected format: YYYY-MM-DDTHH:MM:SS
    // e.g. "2026-07-17T20:00:46.447583+00:00" or "2026-03-18T19:11:09.667767Z"
    if (str.len < 19) return error.InvalidTimestampFormat;
    const year = std.fmt.parseInt(i32, str[0..4], 10) catch return error.InvalidTimestampFormat;
    if (str[4] != '-') return error.InvalidTimestampFormat;
    const month = std.fmt.parseInt(i32, str[5..7], 10) catch return error.InvalidTimestampFormat;
    if (str[7] != '-') return error.InvalidTimestampFormat;
    const day = std.fmt.parseInt(i32, str[8..10], 10) catch return error.InvalidTimestampFormat;
    if (str[10] != 'T' and str[10] != ' ') return error.InvalidTimestampFormat;
    const hour = std.fmt.parseInt(i32, str[11..13], 10) catch return error.InvalidTimestampFormat;
    if (str[13] != ':') return error.InvalidTimestampFormat;
    const minute = std.fmt.parseInt(i32, str[14..16], 10) catch return error.InvalidTimestampFormat;
    if (str[16] != ':') return error.InvalidTimestampFormat;
    const second = std.fmt.parseInt(i32, str[17..19], 10) catch return error.InvalidTimestampFormat;

    var days: i64 = 0;
    var y: i32 = 1970;
    while (y < year) : (y += 1) {
        days += if (isLeapYear(y)) @as(i64, 366) else @as(i64, 365);
    }
    while (y > year) : (y -= 1) {
        days -= if (isLeapYear(y - 1)) @as(i64, 366) else @as(i64, 365);
    }

    const month_days = [_]i64{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    var m: usize = 0;
    while (m < @as(usize, @intCast(month - 1))) : (m += 1) {
        if (m == 1 and isLeapYear(year)) {
            days += 29;
        } else {
            days += month_days[m];
        }
    }

    days += (day - 1);

    const seconds_in_day: i64 = 86400;
    const seconds_in_hour: i64 = 3600;
    const seconds_in_minute: i64 = 60;

    const total_seconds = days * seconds_in_day + @as(i64, hour) * seconds_in_hour + @as(i64, minute) * seconds_in_minute + second;

    var fraction: f64 = 0.0;
    var index: usize = 19;
    if (index < str.len and str[index] == '.') {
        index += 1;
        const start = index;
        while (index < str.len and std.ascii.isDigit(str[index])) : (index += 1) {}
        if (index > start) {
            fraction = std.fmt.parseFloat(f64, str[start - 1 .. index]) catch 0.0;
        }
    }

    var tz_offset_seconds: i64 = 0;
    if (index < str.len) {
        const tz_char = str[index];
        if (tz_char == 'Z') {
            // UTC
        } else if (tz_char == '+' or tz_char == '-') {
            if (str.len >= index + 3) {
                const tz_hours = std.fmt.parseInt(i64, str[index + 1 .. index + 3], 10) catch 0;
                var tz_mins: i64 = 0;
                if (str.len >= index + 6 and str[index + 3] == ':') {
                    tz_mins = std.fmt.parseInt(i64, str[index + 4 .. index + 6], 10) catch 0;
                } else if (str.len >= index + 5) {
                    tz_mins = std.fmt.parseInt(i64, str[index + 3 .. index + 5], 10) catch 0;
                }
                const offset = tz_hours * 3600 + tz_mins * 60;
                if (tz_char == '+') {
                    tz_offset_seconds = offset;
                } else {
                    tz_offset_seconds = -offset;
                }
            }
        }
    }

    return @as(f64, @floatFromInt(total_seconds - tz_offset_seconds)) + fraction;
}

pub fn findMostRecentPriceIndex(times: []const f64, target_time: f64) ?usize {
    if (times.len == 0) return null;
    if (target_time < times[0]) return null;

    var low: usize = 0;
    var high: usize = times.len - 1;
    var ans: ?usize = null;

    while (low <= high) {
        const mid = low + (high - low) / 2;
        if (times[mid] <= target_time) {
            ans = mid;
            low = mid + 1;
        } else {
            if (mid == 0) break;
            high = mid - 1;
        }
    }
    return ans;
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.readToEndAlloc(allocator, std.math.maxInt(usize));
}

pub const AlignedPair = struct {
    kalshi_idx: usize,
    btc_idx: usize,
};

pub fn parseAndAlign(
    allocator: std.mem.Allocator,
    kalshi_path: []const u8,
    btc_path: []const u8,
    cfg: FeatureConfig,
) !AlignedDataset {
    std.debug.print("Reading Kalshi dataset from: {s}...\n", .{kalshi_path});
    const kalshi_bytes = try readFileAlloc(allocator, kalshi_path);
    defer allocator.free(kalshi_bytes);

    std.debug.print("Reading BTC dataset from: {s}...\n", .{btc_path});
    const btc_bytes = try readFileAlloc(allocator, btc_path);
    defer allocator.free(btc_bytes);

    std.debug.print("Parsing Kalshi JSON...\n", .{});
    var kalshi_parsed = try std.json.parseFromSlice(KalshiDataset, allocator, kalshi_bytes, .{
        .ignore_unknown_fields = true,
    });
    defer kalshi_parsed.deinit();

    std.debug.print("Parsing BTC price JSON...\n", .{});
    var btc_parsed = try std.json.parseFromSlice(BtcPriceDataset, allocator, btc_bytes, .{
        .ignore_unknown_fields = true,
    });
    defer btc_parsed.deinit();

    const kalshi_data = kalshi_parsed.value;
    const btc_data = btc_parsed.value;

    std.debug.print("Loaded {d} Kalshi rows, {d} BTC price points.\n", .{
        kalshi_data.training_rows.len,
        btc_data.y_rows.len,
    });

    std.debug.print("Converting BTC timestamps to Unix seconds...\n", .{});
    var btc_times = try allocator.alloc(f64, btc_data.y_rows.len);
    defer allocator.free(btc_times);

    for (btc_data.y_rows, 0..) |row, i| {
        btc_times[i] = try parseIso8601ToSeconds(row.timestamp);
    }

    std.debug.print("Aligning datasets...\n", .{});
    var aligned_indices = std.array_list.Managed(AlignedPair).init(allocator);
    defer aligned_indices.deinit();

    for (kalshi_data.training_rows, 0..) |row, i| {
        const t = try parseIso8601ToSeconds(row.timestamp);
        // Add lookahead seconds to get the target timestamp
        const target_time = t + @as(f64, @floatFromInt(row.lookahead_seconds));
        if (findMostRecentPriceIndex(btc_times, target_time)) |btc_idx| {
            try aligned_indices.append(.{ .kalshi_idx = i, .btc_idx = btc_idx });
        }
    }

    const num_aligned = aligned_indices.items.len;
    std.debug.print("Successfully aligned {d} data points.\n", .{num_aligned});

    if (num_aligned == 0) return error.NoAlignedData;

    const num_features = cfg.countFeatures();
    if (num_features == 0) return error.NoFeaturesSelected;

    std.debug.print("Creating DataObject tensors... Features: {d}\n", .{num_features});
    var x_tensor = try trix.DataObject.init(allocator, &[_]usize{ num_aligned, num_features }, .f32);
    errdefer x_tensor.deinit();

    var y_tensor = try trix.DataObject.init(allocator, &[_]usize{ num_aligned, 1 }, .f32);
    errdefer y_tensor.deinit();

    for (aligned_indices.items, 0..) |pair, row_idx| {
        const k_row = kalshi_data.training_rows[pair.kalshi_idx];
        const b_row = btc_data.y_rows[pair.btc_idx];

        const base = row_idx * num_features;
        var f_idx: usize = 0;

        if (cfg.probability) {
            x_tensor.values.items[base + f_idx] = k_row.probability;
            f_idx += 1;
        }
        if (cfg.lookahead_seconds) {
            x_tensor.values.items[base + f_idx] = @as(f32, @floatFromInt(k_row.lookahead_seconds));
            f_idx += 1;
        }

        y_tensor.values.items[row_idx] = b_row.price;
    }

    return AlignedDataset{
        .allocator = allocator,
        .x_tensor = x_tensor,
        .y_tensor = y_tensor,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip executable name
    _ = args.next();

    var kalshi_path: []const u8 = "kalshi_training_120d_dataset.json";
    var btc_path: []const u8 = "btc_price_120d_dataset.json";
    var cfg = FeatureConfig{};

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--kalshi")) {
            kalshi_path = args.next() orelse {
                std.debug.print("Error: --kalshi requires a path argument\n", .{});
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, arg, "--btc")) {
            btc_path = args.next() orelse {
                std.debug.print("Error: --btc requires a path argument\n", .{});
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, arg, "--no-prob")) {
            cfg.probability = false;
        } else if (std.mem.eql(u8, arg, "--no-lookahead")) {
            cfg.lookahead_seconds = false;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print(
                \\Crypto Dataset Parser CLI
                \\Usage: crypto-parser [options]
                \\
                \\Options:
                \\  --kalshi <path>      Path to Kalshi JSON dataset (default: kalshi_training_120d_dataset.json)
                \\  --btc <path>         Path to BTC price JSON dataset (default: btc_price_120d_dataset.json)
                \\  --no-prob            Exclude 'probability' feature
                \\  --no-lookahead       Exclude 'lookahead_seconds' feature
                \\  -h, --help           Show this help text
                \\
            , .{});
            return;
        } else {
            std.debug.print("Error: Unknown argument: {s}\n", .{arg});
            return error.InvalidArgument;
        }
    }

    var dataset = parseAndAlign(allocator, kalshi_path, btc_path, cfg) catch |err| {
        std.debug.print("Failed to parse and align datasets: {any}\n", .{err});
        return err;
    };
    defer dataset.deinit();

    std.debug.print("\n--- Parsing Summary ---\n", .{});
    std.debug.print("X Tensor Shape: {any}\n", .{dataset.x_tensor.shape.?.items});
    std.debug.print("Y Tensor Shape: {any}\n", .{dataset.y_tensor.shape.?.items});
    std.debug.print("First 3 aligned X feature values:\n", .{});
    var r: usize = 0;
    while (r < @min(@as(usize, 3), dataset.x_tensor.shape.?.items[0])) : (r += 1) {
        std.debug.print("  Row {d}: ", .{r});
        var c: usize = 0;
        const num_f = dataset.x_tensor.shape.?.items[1];
        while (c < num_f) : (c += 1) {
            std.debug.print("{d:.6}  ", .{dataset.x_tensor.get(&[_]usize{ r, c })});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("First 3 aligned Y target prices:\n", .{});
    r = 0;
    while (r < @min(@as(usize, 3), dataset.y_tensor.shape.?.items[0])) : (r += 1) {
        std.debug.print("  Row {d}: {d:.2}\n", .{r, dataset.y_tensor.get(&[_]usize{ r, 0 })});
    }
    std.debug.print("-----------------------\n", .{});
}

test "parseIso8601ToSeconds parses UTC formats correctly" {
    const t1 = try parseIso8601ToSeconds("2026-03-18T19:11:09Z");
    try std.testing.expectApproxEqAbs(@as(f64, 1773861069.0), t1, 1e-3);

    const t2 = try parseIso8601ToSeconds("2026-07-17T20:00:46.447583+00:00");
    try std.testing.expectApproxEqAbs(@as(f64, 1784318446.447583), t2, 1e-3);
}

test "findMostRecentPriceIndex performs correct alignment binary search" {
    const btc_times = [_]f64{ 10.0, 20.0, 30.0 };

    // Query before range
    try std.testing.expectEqual(@as(?usize, null), findMostRecentPriceIndex(&btc_times, 5.0));

    // Query exact match
    try std.testing.expectEqual(@as(?usize, 0), findMostRecentPriceIndex(&btc_times, 10.0));
    try std.testing.expectEqual(@as(?usize, 1), findMostRecentPriceIndex(&btc_times, 20.0));

    // Query between elements
    try std.testing.expectEqual(@as(?usize, 0), findMostRecentPriceIndex(&btc_times, 15.0));
    try std.testing.expectEqual(@as(?usize, 1), findMostRecentPriceIndex(&btc_times, 25.0));

    // Query after range
    try std.testing.expectEqual(@as(?usize, 2), findMostRecentPriceIndex(&btc_times, 35.0));
}

test "FeatureConfig toggles features correctly" {
    var cfg = FeatureConfig{};
    try std.testing.expectEqual(@as(usize, 2), cfg.countFeatures());

    cfg.probability = false;
    try std.testing.expectEqual(@as(usize, 1), cfg.countFeatures());

    cfg.lookahead_seconds = false;
    try std.testing.expectEqual(@as(usize, 0), cfg.countFeatures());
}
