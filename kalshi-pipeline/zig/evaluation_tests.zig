const std = @import("std");
const parser = @import("crypto_dataset");

test "crypto dataset parser aligns Kalshi rows with BTC prices" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const kalshi_json =
        \\{
        \\  "crypto": "BTC",
        \\  "requested_days": 120,
        \\  "window_start": "2026-03-19T20:00:00Z",
        \\  "window_end": "2026-07-17T20:00:00Z",
        \\  "training_rows": [
        \\    {
        \\      "ticker": "KXBTC15M",
        \\      "timestamp": "2026-07-17T20:00:46Z",
        \\      "probability": 0.55,
        \\      "lookahead_seconds": 0
        \\    }
        \\  ]
        \\}
    ;

    const btc_json =
        \\{
        \\  "crypto": "BTC",
        \\  "quote_currency": "USD",
        \\  "product_id": "BTC-USD",
        \\  "requested_days": 120,
        \\  "window_start": "2026-03-18T19:00:00Z",
        \\  "window_end": "2026-07-17T20:00:00Z",
        \\  "price_source": "candles",
        \\  "source": "Coinbase",
        \\  "target_definition": "candle close price",
        \\  "row_count": 2,
        \\  "y_rows": [
        \\    {
        \\      "timestamp": "2026-07-17T20:00:00Z",
        \\      "price": 63999.99
        \\    },
        \\    {
        \\      "timestamp": "2026-07-17T20:01:00Z",
        \\      "price": 64005.50
        \\    }
        \\  ]
        \\}
    ;

    if (@hasDecl(std.testing, "io")) {
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "kalshi.json", .data = kalshi_json });
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "btc.json", .data = btc_json });
    } else {
        try tmp.dir.writeFile(.{ .sub_path = "kalshi.json", .data = kalshi_json });
        try tmp.dir.writeFile(.{ .sub_path = "btc.json", .data = btc_json });
    }

    const kalshi_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/kalshi.json", .{tmp.sub_path});
    defer allocator.free(kalshi_path);
    const btc_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/btc.json", .{tmp.sub_path});
    defer allocator.free(btc_path);

    var dataset = try parser.parseAndAlign(allocator, kalshi_path, btc_path, .{});
    defer dataset.deinit();

    try std.testing.expectEqual(@as(usize, 1), dataset.x_tensor.shape.?.items[0]);
    try std.testing.expectEqual(@as(usize, 2), dataset.x_tensor.shape.?.items[1]);
    try std.testing.expectEqual(@as(usize, 1), dataset.y_tensor.shape.?.items[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 63999.99), dataset.y_tensor.values.items[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.55), dataset.x_tensor.get(&[_]usize{ 0, 0 }), 1e-4);
}
