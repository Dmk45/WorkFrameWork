const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const framework_path = b.option(
        []const u8,
        "framework-path",
        "Path to a local ModelWork2 checkout",
    ) orelse "../WorkFrameWork";
    const framework_source = std.fs.path.join(b.allocator, &.{
        framework_path,
        "src",
        "lib.zig",
    }) catch @panic("Failed to construct framework source path");

    const framework_module = b.createModule(.{
        .root_source_file = .{ .cwd_relative = framework_source },
        .target = target,
        .optimize = optimize,
    });

    const crypto_dataset_module = b.createModule(.{
        .root_source_file = b.path("zig/crypto_dataset.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "modelwork2", .module = framework_module }},
    });

    const parser_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/evaluation_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "modelwork2", .module = framework_module },
                .{ .name = "crypto_dataset", .module = crypto_dataset_module },
            },
        }),
    });
    const test_step = b.step("test", "Run Kalshi dataset alignment tests");
    test_step.dependOn(&b.addRunArtifact(parser_tests).step);

    const parser = b.addExecutable(.{
        .name = "crypto-parser",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/crypto_dataset.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "modelwork2", .module = framework_module }},
        }),
    });
    b.installArtifact(parser);

    const run_parser = b.addRunArtifact(parser);
    if (b.args) |args| run_parser.addArgs(args);
    const parse_step = b.step("parse", "Parse and align the generated datasets");
    parse_step.dependOn(&run_parser.step);

    const model = b.addExecutable(.{
        .name = "crypto-model",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/Modelrunt.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "modelwork2", .module = framework_module },
                .{ .name = "crypto_dataset", .module = crypto_dataset_module },
            },
        }),
    });
    b.installArtifact(model);

    const run_model = b.addRunArtifact(model);
    const model_step = b.step("crypto-model", "Train and evaluate the crypto model");
    model_step.dependOn(&run_model.step);
}
