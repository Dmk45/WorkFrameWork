const std = @import("std");
const lib = @import("modelwork2");

fn writeTestFile(dir: anytype, sub_path: []const u8, data: []const u8) !void {
    if (@hasDecl(std.testing, "io")) {
        try dir.writeFile(std.testing.io, .{ .sub_path = sub_path, .data = data });
    } else {
        try dir.writeFile(.{ .sub_path = sub_path, .data = data });
    }
}

fn tmpFilePath(allocator: std.mem.Allocator, tmp: anytype, sub_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, sub_path });
}

fn accessTestFile(dir: anytype, sub_path: []const u8) !void {
    if (@hasDecl(std.testing, "io")) {
        try dir.access(std.testing.io, sub_path, .{});
    } else {
        try dir.access(sub_path, .{});
    }
}

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

test "data pipeline loads csv and exposes rows" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const csv_content =
        \\f1,f2,label
        \\1.0,2.0,0.0
        \\3.0,4.0,1.0
    ;
    try writeTestFile(tmp.dir, "sample.csv", csv_content);

    const sample_path = try tmpFilePath(allocator, tmp, "sample.csv");
    defer allocator.free(sample_path);
    var dataset = try lib.data_pipeline.CsvDataset.loadFromCsv(allocator, sample_path, true);
    defer dataset.deinit();

    try std.testing.expectEqual(@as(usize, 2), dataset.row_count);
    try std.testing.expectEqual(@as(usize, 2), dataset.feature_count);

    const row0 = try dataset.rowFeatures(0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), row0[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try dataset.rowLabel(0), 1e-6);
}

test "data pipeline split and batching are deterministic" {
    const allocator = std.testing.allocator;

    var split = try lib.data_pipeline.splitIndices(allocator, 10, 0.6, 0.2, 42);
    defer split.deinit();

    try std.testing.expectEqual(@as(usize, 6), split.train_indices.items.len);
    try std.testing.expectEqual(@as(usize, 2), split.val_indices.items.len);
    try std.testing.expectEqual(@as(usize, 2), split.test_indices.items.len);

    var iter = try lib.data_pipeline.BatchIterator.init(split.train_indices.items, 4);
    const b1 = iter.next() orelse unreachable;
    const b2 = iter.next() orelse unreachable;
    const b3 = iter.next();

    try std.testing.expectEqual(@as(usize, 4), b1.len);
    try std.testing.expectEqual(@as(usize, 2), b2.len);
    try std.testing.expect(b3 == null);
}

test "data loader supports stratified batches and caching/prefetch" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const csv_content =
        \\f1,f2,label
        \\1,10,0
        \\2,20,0
        \\3,30,1
        \\4,40,1
        \\5,50,0
        \\6,60,1
    ;
    try writeTestFile(tmp.dir, "loader.csv", csv_content);

    const loader_path = try tmpFilePath(allocator, tmp, "loader.csv");
    defer allocator.free(loader_path);
    var ds = try lib.data_pipeline.CsvDataset.loadFromCsv(allocator, loader_path, true);
    defer ds.deinit();

    var split = try lib.data_pipeline.splitIndices(allocator, ds.row_count, 1.0, 0.0, 1);
    defer split.deinit();

    var loader = try lib.data_pipeline.DataLoader.init(
        allocator,
        ds.asDataset(),
        split.train_indices.items,
        .{
            .batch_size = 2,
            .strategy = .stratified,
            .seed = 1,
            .prefetch = true,
            .use_cache = true,
        },
    );
    defer loader.deinit();

    const b1 = (try loader.next()) orelse unreachable;
    try std.testing.expectEqual(@as(usize, 2), b1.batch_len);
    try std.testing.expectEqual(@as(usize, 2), b1.feature_count);
    try std.testing.expectEqual(@as(usize, 2), b1.labels.len);
}

test "scalers fit and transform features" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const csv_content =
        \\f1,f2,label
        \\1,2,0
        \\3,4,1
        \\5,6,0
    ;
    try writeTestFile(tmp.dir, "scale.csv", csv_content);

    const scale_path = try tmpFilePath(allocator, tmp, "scale.csv");
    defer allocator.free(scale_path);
    var ds = try lib.data_pipeline.CsvDataset.loadFromCsv(allocator, scale_path, true);
    defer ds.deinit();

    const idx = [_]usize{ 0, 1, 2 };
    var std_scaler = try lib.data_pipeline.StandardScaler.fit(allocator, ds.asDataset(), &idx);
    defer std_scaler.deinit();
    var mm_scaler = try lib.data_pipeline.MinMaxScaler.fit(allocator, ds.asDataset(), &idx);
    defer mm_scaler.deinit();

    var row = [_]f32{ 3, 4 };
    std_scaler.transform(&row);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), row[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), row[1], 1e-5);

    var row2 = [_]f32{ 3, 4 };
    mm_scaler.transform(&row2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), row2[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), row2[1], 1e-5);
}

test "section 3 core and advanced layer features" {
    const allocator = std.testing.allocator;

    var conv1d = try lib.layers.Conv1DLayer.init(allocator, 1, 2, 3, 1, 1, null, null);
    defer conv1d.deinit();
    var x1d = try lib.maker.zeros(allocator, &[_]usize{ 2, 1, 8 });
    defer x1d.deinit();
    var y1d = try conv1d.forward(allocator, &x1d);
    defer y1d.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 8 }, y1d.shape.?.items);

    var conv2d = try lib.layers.Conv2DLayer.init(allocator, 3, 4, 3, 3, 1, 1, null, null);
    defer conv2d.deinit();
    var x2d = try lib.maker.zeros(allocator, &[_]usize{ 2, 3, 6, 6 });
    defer x2d.deinit();
    var y2d = try conv2d.forward(allocator, &x2d);
    defer y2d.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4, 6, 6 }, y2d.shape.?.items);

    var conv3d = try lib.layers.Conv3DLayer.init(allocator, 2, 3, 3, 3, 3, 1, 1, null, null);
    defer conv3d.deinit();
    var x3d = try lib.maker.zeros(allocator, &[_]usize{ 1, 2, 5, 5, 5 });
    defer x3d.deinit();
    var y3d = try conv3d.forward(allocator, &x3d);
    defer y3d.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3, 5, 5, 5 }, y3d.shape.?.items);

    var pooled_max = try lib.layers.pool2d(allocator, &x2d, 2, 2, .max);
    defer pooled_max.deinit();
    var pooled_avg = try lib.layers.pool2d(allocator, &x2d, 2, 2, .avg);
    defer pooled_avg.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 3, 3 }, pooled_max.shape.?.items);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 3, 3 }, pooled_avg.shape.?.items);

    var flat = try lib.layers.flatten(allocator, &x2d);
    defer flat.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 108 }, flat.shape.?.items);
    try lib.layers.reshapeDynamic(&flat, &[_]usize{ 2, 54, 2 });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 54, 2 }, flat.shape.?.items);

    var dropped = try lib.layers.dropout(allocator, &flat, 0.5, true, 42);
    defer dropped.deinit();
    try std.testing.expectEqual(flat.values.items.len, dropped.values.items.len);

    var norm_in = try lib.maker.zeros(allocator, &[_]usize{ 4, 6 });
    defer norm_in.deinit();
    for (norm_in.values.items, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));
    var bn = try lib.layers.batchNorm(allocator, &norm_in, 1e-5);
    defer bn.deinit();
    var ln = try lib.layers.layerNorm(allocator, &norm_in, 1e-5);
    defer ln.deinit();
    var gn = try lib.layers.groupNorm(allocator, &norm_in, 3, 1e-5);
    defer gn.deinit();
    try std.testing.expectEqual(norm_in.values.items.len, bn.values.items.len);
    try std.testing.expectEqual(norm_in.values.items.len, ln.values.items.len);
    try std.testing.expectEqual(norm_in.values.items.len, gn.values.items.len);

    var lstm = try lib.layers.LSTMCell.init(allocator, 6, 4, null, null);
    defer lstm.deinit();
    var gru = try lib.layers.GRUCell.init(allocator, 6, 4, null, null);
    defer gru.deinit();
    var x_step = try lib.maker.zeros(allocator, &[_]usize{ 2, 6 });
    defer x_step.deinit();
    var h_prev = try lib.maker.zeros(allocator, &[_]usize{ 2, 4 });
    defer h_prev.deinit();
    var c_prev = try lib.maker.zeros(allocator, &[_]usize{ 2, 4 });
    defer c_prev.deinit();
    var lstm_out = try lstm.forward(allocator, &x_step, &h_prev, &c_prev);
    defer lstm_out.h.deinit();
    defer lstm_out.c.deinit();
    var gru_out = try gru.forward(allocator, &x_step, &h_prev);
    defer gru_out.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4 }, lstm_out.h.shape.?.items);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4 }, gru_out.shape.?.items);

    var emb = try lib.layers.EmbeddingLayer.init(allocator, 16, 8);
    defer emb.deinit();
    var emb_out = try emb.forward(allocator, &[_]usize{ 1, 2, 3, 4 });
    defer emb_out.deinit();
    emb.addPositionalEncoding(&emb_out);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 8 }, emb_out.shape.?.items);

    var attn = try lib.layers.MultiHeadSelfAttention.init(2, 8);
    var attn_out = try attn.forward(allocator, &emb_out, &emb_out, &emb_out);
    defer attn_out.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 8 }, attn_out.shape.?.items);

    var res_added = try lib.layers.residualAdd(allocator, &attn_out, &attn_out);
    defer res_added.deinit();
    try std.testing.expectEqual(attn_out.values.items.len, res_added.values.items.len);

    var d1 = try lib.maker.zeros(allocator, &[_]usize{ 2, 3 });
    defer d1.deinit();
    var d2 = try lib.maker.zeros(allocator, &[_]usize{ 2, 4 });
    defer d2.deinit();
    var dense_skip = try lib.layers.denseSkipConcat(allocator, &[_]*lib.matrix.DataObject{ &d1, &d2 });
    defer dense_skip.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 7 }, dense_skip.shape.?.items);
}

test "section 3 composition and introspection features" {
    const allocator = std.testing.allocator;

    var seq = try lib.layers.Sequential.init(allocator);
    defer seq.deinit();
    try seq.addLinear(5, 4, "relu", null, null);
    try seq.addLinear(4, 3, "none", null, null);
    var seq_in = try lib.maker.zeros(allocator, &[_]usize{ 2, 5 });
    defer seq_in.deinit();
    var seq_out = try seq.forward(allocator, &seq_in);
    defer seq_out.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, seq_out.shape.?.items);

    var branch1 = try lib.layers.LinearLayer.init(allocator, 5, 2, "relu", null, null);
    defer branch1.deinit();
    var branch2 = try lib.layers.LinearLayer.init(allocator, 5, 3, "relu", null, null);
    defer branch2.deinit();
    var branched = try lib.layers.branchForward(allocator, &seq_in, &[_]*lib.layers.LinearLayer{ branch1, branch2 });
    defer branched.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 5 }, branched.shape.?.items);

    var rb = try lib.layers.ResidualBlock.init(allocator, 5, 5, 5, false, null, null);
    defer rb.deinit();
    var rb_out = try rb.forward(allocator, &seq_in);
    defer rb_out.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 5 }, rb_out.shape.?.items);

    var inc = try lib.layers.InceptionModule.init(allocator, 5, 2, 2, 2, 2, null, null);
    defer inc.deinit();
    var inc_out = try inc.forward(allocator, &seq_in);
    defer inc_out.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 8 }, inc_out.shape.?.items);

    const linear_stats = lib.layers.linearLayerStats(branch1);
    try std.testing.expectEqualStrings("LinearLayer", linear_stats.name);
    try std.testing.expect(linear_stats.parameter_count > 0);

    var conv2d = try lib.layers.Conv2DLayer.init(allocator, 1, 2, 3, 3, 1, 1, null, null);
    defer conv2d.deinit();
    const conv_stats = lib.layers.conv2dLayerStats(conv2d);
    try std.testing.expectEqualStrings("Conv2DLayer", conv_stats.name);
    try std.testing.expect(conv_stats.parameter_count > 0);
}

test "section 4 autodiff tape and detach utilities" {
    const allocator = std.testing.allocator;
    var a = try lib.maker.zeros(allocator, &[_]usize{ 1, 3 });
    defer a.deinit();
    var b = try lib.maker.zeros(allocator, &[_]usize{ 1, 3 });
    defer b.deinit();
    a.enableGrad();
    b.enableGrad();
    try a.ensureGradValue();
    try b.ensureGradValue();
    for (a.values.items, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));
    for (b.values.items, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 2));
    var out = try lib.core_math.add(allocator, &a, &b);
    defer out.deinit();
    out.enableGrad();
    try out.ensureGradValue();
    @memset(out.grad_value.?.items, 1.0);
    @memset(out.values.items, 1.0);

    var tape = lib.autodiff.Tape.init(allocator);
    defer tape.deinit();
    const a_id = try tape.registerTensor(&a);
    const b_id = try tape.registerTensor(&b);
    const out_id = try tape.registerTensor(&out);
    try tape.record(.Add, &[_]usize{ a_id, b_id }, out_id, .{ .Add = {} });
    try tape.backward();

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), a.grad_value.?.items[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.grad_value.?.items[2], 1e-6);

    var detached = try lib.autodiff.detach(allocator, &out);
    defer detached.deinit();
    lib.autodiff.stopGradient(&detached);
    try std.testing.expect(!detached.grad);
}

test "section 5 optimizers and schedules" {
    const allocator = std.testing.allocator;
    var p = try lib.maker.zeros(allocator, &[_]usize{2});
    defer p.deinit();
    p.enableGrad();
    try p.ensureGradValue();
    p.values.items[0] = 1.0;
    p.values.items[1] = -1.0;
    p.grad_value.?.items[0] = 0.5;
    p.grad_value.?.items[1] = -0.5;

    var sgd = lib.grad.SGD.init(allocator, 0.1, 0.9, false, 0.0);
    defer sgd.deinit();
    try sgd.step(&p);
    try std.testing.expect(p.values.items[0] < 1.0);

    p.grad_value.?.items[0] = 0.25;
    p.grad_value.?.items[1] = -0.25;
    var rms = lib.grad.RMSprop.init(allocator, 0.01, 0.99, 1e-8, 0.0);
    defer rms.deinit();
    try rms.step(&p);

    p.grad_value.?.items[0] = 0.25;
    p.grad_value.?.items[1] = -0.25;
    var ada = lib.grad.AdaGrad.init(allocator, 0.01, 1e-8, 0.0);
    defer ada.deinit();
    try ada.step(&p);

    const step = lib.grad.StepLR{ .step_size = 2, .gamma = 0.5 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), step.step(0.1, 2), 1e-6);
    const exp = lib.grad.ExponentialLR{ .gamma = 0.9 };
    try std.testing.expect(exp.step(0.1, 3) < 0.1);
    const warm = lib.grad.WarmupLR{ .warmup_epochs = 4, .target_lr = 0.2 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), warm.step(0), 1e-6);
}

test "section 6 additional losses" {
    const allocator = std.testing.allocator;
    var y_pred = try lib.maker.zeros(allocator, &[_]usize{4});
    defer y_pred.deinit();
    var y_true = try lib.maker.zeros(allocator, &[_]usize{4});
    defer y_true.deinit();
    @memcpy(y_pred.values.items, &[_]f32{ 0.9, 0.2, 0.8, 0.1 });
    @memcpy(y_true.values.items, &[_]f32{ 1.0, 0.0, 1.0, 0.0 });
    y_pred.enableGrad();

    const bce = try lib.grad.binaryCrossEntropy(&y_pred, &y_true);
    const l1 = try lib.grad.l1Loss(&y_pred, &y_true);
    const huber = try lib.grad.smoothL1Loss(&y_pred, &y_true, 1.0);
    const hinge = try lib.grad.hingeLoss(&y_pred, &y_true);
    try std.testing.expect(bce > 0.0);
    try std.testing.expect(l1 > 0.0);
    try std.testing.expect(huber > 0.0);
    try std.testing.expect(hinge >= 0.0);
}

test "section 7 trainer tracks history and checkpoint" {
    const allocator = std.testing.allocator;
    var net = try lib.layers.NeuralNetwork.init(allocator);
    defer net.deinit();
    const linear = try lib.layers.LinearLayer.init(allocator, 2, 1, "none", null, null);
    try net.add(linear);
    var opt = try lib.grad.Adam.init(allocator, 0.01, 0.9, 0.999, 1e-8);
    defer opt.deinit();
    var trainer = try lib.trainer.Trainer.init(allocator, &net, &opt, .{ .epochs = 2, .clip_grad_norm = 1.0 });
    defer trainer.deinit();

    var x = try lib.maker.zeros(allocator, &[_]usize{ 2, 2 });
    defer x.deinit();
    var y = try lib.maker.zeros(allocator, &[_]usize{ 2, 1 });
    defer y.deinit();
    x.values.items[0] = 1.0;
    x.values.items[1] = 2.0;
    x.values.items[2] = 2.0;
    x.values.items[3] = 3.0;
    y.values.items[0] = 1.0;
    y.values.items[1] = 2.0;

    try trainer.fit(&x, &y, &x, &y);
    try std.testing.expectEqual(@as(usize, 2), trainer.history.epochs.items.len);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const ckpt_path = try tmpFilePath(allocator, tmp, "ckpt.txt");
    defer allocator.free(ckpt_path);
    try trainer.saveCheckpoint(ckpt_path);
    try accessTestFile(tmp.dir, "ckpt.txt");
}

test "section 10 tensor utility operations" {
    const allocator = std.testing.allocator;
    var a = try lib.maker.zeros(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    var b = try lib.maker.zeros(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();
    for (a.values.items, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));
    for (b.values.items, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 5));

    var cat = try lib.core_math.concatenate(allocator, &[_]*lib.matrix.DataObject{ &a, &b }, 1);
    defer cat.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4 }, cat.shape.?.items);

    var st = try lib.core_math.stack(allocator, &[_]*lib.matrix.DataObject{ &a, &b });
    defer st.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 2 }, st.shape.?.items);

    var sq_src = try lib.maker.zeros(allocator, &[_]usize{ 1, 2, 2 });
    defer sq_src.deinit();
    var squeezed = try lib.core_math.squeeze(allocator, &sq_src, 0);
    defer squeezed.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, squeezed.shape.?.items);
    var unsq = try lib.core_math.unsqueeze(allocator, &squeezed, 0);
    defer unsq.deinit();
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 2 }, unsq.shape.?.items);

    var parts = try lib.core_math.split(allocator, &cat, 2, 1);
    defer {
        for (parts.items) |*p| p.deinit();
        parts.deinit();
    }
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
}
