const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_filter = b.option([]const u8, "test-filter", "Filter tests by name");

    const framework_module = b.addModule("modelwork2", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // =========================
    // Examples
    // =========================

    const comprehensive_example = b.addExecutable(.{
        .name = "example-comprehensive",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_comprehensive.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "modelwork2", .module = framework_module }},
        }),
    });
    b.installArtifact(comprehensive_example);

    const layers_example = b.addExecutable(.{
        .name = "example-layers",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_layers.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "modelwork2", .module = framework_module }},
        }),
    });
    b.installArtifact(layers_example);

    const run_comprehensive = b.addRunArtifact(comprehensive_example);
    const run_layers = b.addRunArtifact(layers_example);

    const run_step = b.step("run", "Run all examples");
    run_step.dependOn(&run_comprehensive.step);
    run_step.dependOn(&run_layers.step);

    // =========================
    // Tests
    // =========================

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/framework_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "modelwork2", .module = framework_module }},
        }),
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const modelrunt_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/Modelrunt.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "modelwork2", .module = framework_module }},
        }),
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_modelrunt_tests = b.addRunArtifact(modelrunt_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_modelrunt_tests.step);

    // =========================
    // Docs
    // =========================

    const docs_cmd = b.addSystemCommand(&[_][]const u8{
        b.graph.zig_exe,
        "test",
        "src/lib.zig",
        "--cache-dir",
        b.cache_root.path orelse ".zig-cache",
        "--global-cache-dir",
        b.graph.global_cache_root.path orelse ".zig-cache",
        "-femit-docs",
        "--name",
        "modelwork2-docs",
    });

    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&docs_cmd.step);
}
