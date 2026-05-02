const std = @import("std");

pub const DType = enum {
    f32,
    f64,
    i32,
    i64,
};

fn getRank(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .array => |info| 1 + getRank(info.child),
        else => 0,
    };
}

pub fn shape(comptime T: type) [getRank(T)]usize {
    const rank = comptime getRank(T);
    var res: [rank]usize = undefined;
    var CurrentType = T;
    inline for (0..rank) |i| {
        const info = @typeInfo(CurrentType).array;
        res[i] = info.len;
        CurrentType = info.child;
    }
    return res;
}

pub const DataObject = struct {
    allocator: std.mem.Allocator,
    values: std.ArrayList(f32),
    shape: ?std.ArrayList(usize) = null,
    strides: ?std.ArrayList(usize) = null,
    dtype: DType,
    grad_value: ?std.ArrayList(f32) = null,
    attributes: ?std.StringHashMap(f32) = null,
    grad: bool = false,

    pub fn init(allocator: std.mem.Allocator, dims: []const usize, dtype: DType) !DataObject {
        // 1. Calculate total size and strides
        const strides = try allocator.alloc(usize, dims.len);
        var total_size: usize = 1;

        var i: usize = dims.len;
        while (i > 0) {
            i -= 1;
            strides[i] = total_size;
            total_size *= dims[i];
        }

        var shape_list = try std.ArrayList(usize).initCapacity(allocator, dims.len);
        for (dims) |sh| try shape_list.append(sh);

        var strides_list = try std.ArrayList(usize).initCapacity(allocator, dims.len);
        for (strides) |st| try strides_list.append(st);

        var values = try std.ArrayList(f32).initCapacity(allocator, total_size);
        try values.resize(total_size);

        allocator.free(strides); // since we copied to list

        return .{
            .allocator = allocator,
            .values = values,
            .shape = shape_list,
            .strides = strides_list,
            .dtype = dtype,
        };
    }
    pub fn get(self: DataObject, coords: []const usize) f32 {
        var index: usize = 0;
        for (coords, 0..) |c, i| {
            if (i >= self.strides.?.items.len) break;
            index += c * self.strides.?.items[i];
        }
        return self.values.items[index];
    }

    pub fn remove(self: *DataObject, index: usize) !f32 {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        return self.values.orderedRemove(index);
    }

    pub fn set(self: *DataObject, index: usize, value: f32) !void {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        self.values.items[index] = value;
    }

    // Fast remove (does not preserve order)
    pub fn swapRemove(self: *DataObject, index: usize) !f32 {
        if (index >= self.values.items.len)
            return error.IndexOutOfBounds;

        return self.values.swapRemove(index);
    }

    pub fn getValues(self: DataObject) []f32 {
        return self.values.items;
    }

    pub fn deinit(self: *DataObject) void {
        if (self.shape) |*s| s.deinit();
        if (self.strides) |*s| s.deinit();
        self.values.deinit();
        if (self.grad_value) |*g| g.deinit();
        if (self.attributes) |*a| a.deinit();
    }

    pub fn add(self: *DataObject, value: f32) !void {
        try self.values.append(value);
    }

    fn ensureShape(self: *DataObject) !void {
        if (self.shape == null) {
            self.shape = try std.ArrayList(usize).initCapacity(self.allocator, 4);
        }
    }

    pub fn addDim(self: *DataObject, dim: usize) !void {
        try self.ensureShape();
        try self.shape.?.append(dim);
    }

    pub fn enableGrad(self: *DataObject) void {
        self.grad = true;
    }

    pub fn ensureGradValue(self: *DataObject) !void {
        if (self.grad_value == null) {
            self.grad_value = try std.ArrayList(f32).initCapacity(self.allocator, self.values.items.len);
            try self.grad_value.?.resize(self.values.items.len);
            @memset(self.grad_value.?.items, 0.0);
        }
    }

    pub fn addGrad(self: *DataObject, value: f32) !void {
        try self.ensureGradValue();
        try self.grad_value.?.append(value);
    }

    fn ensureAttributes(self: *DataObject) void {
        if (self.attributes == null) {
            self.attributes = std.StringHashMap(f32).init(self.allocator);
        }
    }

    pub fn setAttr(self: *DataObject, key: []const u8, value: f32) !void {
        self.ensureAttributes();
        try self.attributes.?.put(key, value);
    }

    pub fn getAttr(self: *DataObject, key: []const u8) ?f32 {
        if (self.attributes == null)
            return null;

        return self.attributes.?.get(key);
    }

    pub fn print(self: *const DataObject) void {
        std.debug.print("DataObject\n", .{});

        std.debug.print("dtype: {any}\n", .{self.dtype});

        std.debug.print("values: {any}\n", .{self.values.items});

        if (self.shape) |s| {
            std.debug.print("shape: {any}\n", .{s.items});
        } else {
            std.debug.print("shape: <none>\n", .{});
        }

        std.debug.print("grad enabled: {}\n", .{self.grad});

        if (self.grad_value) |g| {
            std.debug.print("grad values: {any}\n", .{g.items});
        } else {
            std.debug.print("grad values: <none>\n", .{});
        }

        if (self.attributes) |attrs| {
            var it = attrs.iterator();
            while (it.next()) |entry| {
                std.debug.print("attr {s}: {d}\n", .{
                    entry.key_ptr.*,
                    entry.value_ptr.*,
                });
            }
        }
    }
};
