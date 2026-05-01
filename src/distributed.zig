const std = @import("std");
const trix = @import("matrix.zig");
const layers = @import("layers.zig");
const grad = @import("grad.zig");

/// Data parallelism for distributed training
pub const DataParallel = struct {
    world_size: usize,
    rank: usize,
    device_id: usize,
    local_models: []layers.NeuralNetwork,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, world_size: usize, rank: usize, base_model: layers.NeuralNetwork) !DataParallel {
        var local_models = try allocator.alloc(layers.NeuralNetwork, world_size);
        for (0..world_size) |i| {
            local_models[i] = try base_model.clone(allocator);
        }
        
        return .{
            .world_size = world_size,
            .rank = rank,
            .device_id = rank % world_size,
            .local_models = local_models,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DataParallel) void {
        for (self.local_models) |*model| {
            model.deinit();
        }
        self.allocator.free(self.local_models);
    }
    
    pub fn scatterData(self: *DataParallel, data: *trix.DataObject) ![]trix.DataObject {
        const batch_size = data.shape.?.items[0];
        const samples_per_device = batch_size / self.world_size;
        const remainder = batch_size % self.world_size;
        
        var scattered = try self.allocator.alloc(trix.DataObject, self.world_size);
        
        var offset: usize = 0;
        for (0..self.world_size) |device| {
            const device_batch_size = samples_per_device + if (device < remainder) 1 else 0;
            
            if (device_batch_size > 0) {
                const shape = data.shape.?.items;
                var device_shape = try self.allocator.alloc(usize, shape.len);
                std.mem.copy(usize, device_shape, shape);
                device_shape[0] = device_batch_size;
                
                scattered[device] = try trix.DataObject.init(self.allocator, device_shape, data.dtype);
                
                // Copy relevant portion of data
                const start_idx = offset * shape[1]; // Assuming 2D data
                const end_idx = (offset + device_batch_size) * shape[1];
                std.mem.copy(f32, scattered[device].values.items, data.values.items[start_idx..end_idx]);
                
                self.allocator.free(device_shape);
            }
            
            offset += device_batch_size;
        }
        
        return scattered;
    }
    
    pub fn gatherGradients(self: *DataParallel) !void {
        // Simplified gradient gathering - in practice would use NCCL or similar
        if (self.rank == 0) {
            // Root process averages gradients from all devices
            for (0..self.local_models[0].layers.items.len) |layer_idx| {
                const layer = &self.local_models[0].layers.items[layer_idx];
                
                // Accumulate gradients from all devices
                for (1..self.world_size) |device| {
                    const device_layer = &self.local_models[device].layers.items[layer_idx];
                    for (0..layer.weights.values.items.len) |i| {
                        layer.weights.grad_value.?.items[i] += device_layer.weights.grad_value.?.items[i];
                        layer.bias.grad_value.?.items[i] += device_layer.bias.grad_value.?.items[i];
                    }
                }
                
                // Average gradients
                const scale = 1.0 / @as(f32, @floatFromInt(self.world_size));
                for (0..layer.weights.values.items.len) |i| {
                    layer.weights.grad_value.?.items[i] *= scale;
                    layer.bias.grad_value.?.items[i] *= scale;
                }
            }
        }
        
        // Broadcast averaged gradients back to all devices
        if (self.rank == 0) {
            for (1..self.world_size) |device| {
                for (0..self.local_models[0].layers.items.len) |layer_idx| {
                    const root_layer = &self.local_models[0].layers.items[layer_idx];
                    const device_layer = &self.local_models[device].layers.items[layer_idx];
                    
                    std.mem.copy(f32, device_layer.weights.grad_value.?.items, root_layer.weights.grad_value.?.items);
                    std.mem.copy(f32, device_layer.bias.grad_value.?.items, root_layer.bias.grad_value.?.items);
                }
            }
        }
    }
    
    pub fn synchronizeParameters(self: *DataParallel) !void {
        // Ensure all devices have the same parameters
        if (self.rank == 0) {
            for (1..self.world_size) |device| {
                for (0..self.local_models[0].layers.items.len) |layer_idx| {
                    const root_layer = &self.local_models[0].layers.items[layer_idx];
                    const device_layer = &self.local_models[device].layers.items[layer_idx];
                    
                    std.mem.copy(f32, device_layer.weights.values.items, root_layer.weights.values.items);
                    std.mem.copy(f32, device_layer.bias.values.items, root_layer.bias.values.items);
                }
            }
        }
    }
};

/// Model parallelism for large models
pub const ModelParallel = struct {
    pipeline_stages: []PipelineStage,
    num_stages: usize,
    rank: usize,
    allocator: std.mem.Allocator,
    
    const PipelineStage = struct {
        layers: []layers.Layer,
        input_buffer: ?trix.DataObject,
        output_buffer: ?trix.DataObject,
        is_busy: bool,
    };
    
    pub fn init(allocator: std.mem.Allocator, model: layers.NeuralNetwork, num_stages: usize, rank: usize) !ModelParallel {
        const layers_per_stage = model.layers.items.len / num_stages;
        var pipeline_stages = try allocator.alloc(PipelineStage, num_stages);
        
        for (0..num_stages) |stage| {
            const start_layer = stage * layers_per_stage;
            const end_layer = if (stage == num_stages - 1) model.layers.items.len else (stage + 1) * layers_per_stage;
            
            pipeline_stages[stage] = .{
                .layers = model.layers.items[start_layer..end_layer],
                .input_buffer = null,
                .output_buffer = null,
                .is_busy = false,
            };
        }
        
        return .{
            .pipeline_stages = pipeline_stages,
            .num_stages = num_stages,
            .rank = rank,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ModelParallel) void {
        for (self.pipeline_stages) |*stage| {
            if (stage.input_buffer) |*buf| buf.deinit();
            if (stage.output_buffer) |*buf| buf.deinit();
        }
        self.allocator.free(self.pipeline_stages);
    }
    
    pub fn forwardPipeline(self: *ModelParallel, input: *trix.DataObject) !trix.DataObject {
        var current_input = input;
        
        for (0..self.num_stages) |stage| {
            const stage_ref = &self.pipeline_stages[stage];
            
            // Process layers in this stage
            for (stage_ref.layers) |*layer| {
                var output = try layer.forward(self.allocator, current_input);
                if (current_input != input) current_input.deinit();
                current_input = output;
            }
            
            // Pass to next stage
            if (stage < self.num_stages - 1) {
                const next_stage = &self.pipeline_stages[stage + 1];
                next_stage.input_buffer = try current_input.clone(self.allocator);
            }
        }
        
        return current_input;
    }
};

/// Gradient synchronization utilities
pub const GradientSync = struct {
    pub const AllReduceOp = enum {
        sum,
        mean,
        max,
        min,
    };
    
    pub fn allReduce(tensors: []trix.DataObject, op: AllReduceOp, allocator: std.mem.Allocator) !void {
        if (tensors.len == 0) return;
        
        switch (op) {
            .sum => {
                // Sum all tensors into the first one
                for (1..tensors.len) |i| {
                    for (0..tensors[0].values.items.len) |j| {
                        tensors[0].values.items[j] += tensors[i].values.items[j];
                    }
                }
                
                // Broadcast result to all tensors
                for (1..tensors.len) |i| {
                    std.mem.copy(f32, tensors[i].values.items, tensors[0].values.items);
                }
            },
            .mean => {
                // Sum first, then divide by count
                try allReduce(tensors, .sum, allocator);
                const scale = 1.0 / @as(f32, @floatFromInt(tensors.len));
                for (tensors) |*tensor| {
                    for (tensor.values.items) |*val| {
                        val.* *= scale;
                    }
                }
            },
            .max => {
                // Find element-wise maximum
                for (0..tensors[0].values.items.len) |j| {
                    var max_val = tensors[0].values.items[j];
                    for (1..tensors.len) |i| {
                        if (tensors[i].values.items[j] > max_val) {
                            max_val = tensors[i].values.items[j];
                        }
                    }
                    
                    for (tensors) |*tensor| {
                        tensor.values.items[j] = max_val;
                    }
                }
            },
            .min => {
                // Find element-wise minimum
                for (0..tensors[0].values.items.len) |j| {
                    var min_val = tensors[0].values.items[j];
                    for (1..tensors.len) |i| {
                        if (tensors[i].values.items[j] < min_val) {
                            min_val = tensors[i].values.items[j];
                        }
                    }
                    
                    for (tensors) |*tensor| {
                        tensor.values.items[j] = min_val;
                    }
                }
            },
        }
    }
    
    pub fn broadcast(tensor: *trix.DataObject, root_rank: usize, world_size: usize, allocator: std.mem.Allocator) ![]trix.DataObject {
        var broadcasted = try allocator.alloc(trix.DataObject, world_size);
        
        for (0..world_size) |rank| {
            broadcasted[rank] = try tensor.clone(allocator);
        }
        
        return broadcasted;
    }
};

/// Distributed sampling for consistent data shuffling
pub const DistributedSampler = struct {
    dataset_size: usize,
    batch_size: usize,
    world_size: usize,
    rank: usize,
    seed: u64,
    indices: []usize,
    current_pos: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, dataset_size: usize, batch_size: usize, world_size: usize, rank: usize, seed: u64) !DistributedSampler {
        var indices = try allocator.alloc(usize, dataset_size);
        for (0..dataset_size) |i| {
            indices[i] = i;
        }
        
        return .{
            .dataset_size = dataset_size,
            .batch_size = batch_size,
            .world_size = world_size,
            .rank = rank,
            .seed = seed,
            .indices = indices,
            .current_pos = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DistributedSampler) void {
        self.allocator.free(self.indices);
    }
    
    pub fn setEpoch(self: *DistributedSampler, epoch: usize) void {
        self.current_pos = 0;
        
        // Shuffle with epoch-dependent seed
        var prng = std.Random.DefaultPrng.init(self.seed + @as(u64, @intCast(epoch)));
        var random = prng.random();
        
        // Fisher-Yates shuffle
        for (1..self.indices.len) |i| {
            const j = random.uintLessThan(usize, i + 1);
            std.mem.swap(usize, &self.indices[i], &self.indices[j]);
        }
    }
    
    pub fn getBatchIndices(self: *DistributedSampler) !?[]usize {
        if (self.current_pos >= self.dataset_size) return null;
        
        const samples_per_rank = (self.dataset_size + self.world_size - 1) / self.world_size;
        const start_pos = self.rank * samples_per_rank;
        const end_pos = @min(start_pos + samples_per_rank, self.dataset_size);
        
        const remaining = end_pos - self.current_pos;
        const actual_batch_size = @min(self.batch_size, remaining);
        
        if (actual_batch_size == 0) return null;
        
        const batch_indices = try self.allocator.alloc(usize, actual_batch_size);
        for (0..actual_batch_size) |i| {
            batch_indices[i] = self.indices[self.current_pos + i];
        }
        
        self.current_pos += actual_batch_size;
        return batch_indices;
    }
};

/// Distributed training coordinator
pub const DistributedTrainer = struct {
    data_parallel: DataParallel,
    sampler: DistributedSampler,
    world_size: usize,
    rank: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, world_size: usize, rank: usize, model: layers.NeuralNetwork, dataset_size: usize, batch_size: usize, seed: u64) !DistributedTrainer {
        const data_parallel = try DataParallel.init(allocator, world_size, rank, model);
        const sampler = try DistributedSampler.init(allocator, dataset_size, batch_size, world_size, rank, seed);
        
        return .{
            .data_parallel = data_parallel,
            .sampler = sampler,
            .world_size = world_size,
            .rank = rank,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DistributedTrainer) void {
        self.data_parallel.deinit();
        self.sampler.deinit();
    }
    
    pub fn trainStep(self: *DistributedTrainer, data: *trix.DataObject, targets: *trix.DataObject, optimizer: anytype) !f32 {
        // Scatter data across devices
        const scattered_data = try self.data_parallel.scatterData(data);
        defer {
            for (scattered_data) |*tensor| tensor.deinit();
            self.allocator.free(scattered_data);
        }
        
        const scattered_targets = try self.data_parallel.scatterData(targets);
        defer {
            for (scattered_targets) |*tensor| tensor.deinit();
            self.allocator.free(scattered_targets);
        }
        
        // Forward pass on local model
        const local_model = &self.data_parallel.local_models[self.data_parallel.rank];
        const predictions = try local_model.forward(self.allocator, &scattered_data[self.data_parallel.rank]);
        defer predictions.deinit();
        
        // Compute loss
        const loss = try grad.meanSquaredError(&predictions, &scattered_targets[self.data_parallel.rank]);
        
        // Gather gradients
        try self.data_parallel.gatherGradients();
        
        // Update parameters (only on rank 0, then broadcast)
        if (self.rank == 0) {
            try local_model.update_parameters(optimizer);
        }
        
        // Synchronize parameters across all devices
        try self.data_parallel.synchronizeParameters();
        
        return loss;
    }
    
    pub fn trainEpoch(self: *DistributedTrainer, dataset: *trix.DataObject, targets: *trix.DataObject, optimizer: anytype, epoch: usize) !f32 {
        self.sampler.setEpoch(epoch);
        
        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;
        
        while (self.sampler.getBatchIndices()) |batch_indices| {
            defer self.allocator.free(batch_indices);
            
            // Extract batch data (simplified - would need proper batching logic)
            const batch_loss = try self.trainStep(dataset, targets, optimizer);
            total_loss += batch_loss;
            num_batches += 1;
        }
        
        return if (num_batches > 0) total_loss / @as(f32, @floatFromInt(num_batches)) else 0.0;
    }
};
