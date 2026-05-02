const std = @import("std");
const lib = @import("src/lib.zig");

test "core modules compile and basic tensor operations run" {
    const allocator = std.testing.allocator;

    var a = try lib.maker.zeros(allocator, &[_]usize{ 1, 3 });
    defer a.deinit();

    var b = try lib.maker.zeros(allocator, &[_]usize{ 1, 3 });
    defer b.deinit();
    b.values.items[0] = 1.0;
    b.values.items[1] = 1.0;
    b.values.items[2] = 1.0;

    var c = try lib.core_math.add(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(@as(usize, 3), c.values.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c.values.items[0], 1e-6);
}
