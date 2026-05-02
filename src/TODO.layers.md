Fix 

    pub fn add(self: *NeuralNetwork, layer: anytype) !void {
        const LayerStruct = @TypeOf(layer);
        
        // Determine which union variant to use based on type
        if (LayerStruct == LinearLayer) {
            try self.layers.append(Layer{ .linear = layer });
        } else if (LayerStruct == Conv1DLayer) {
            try self.layers.append(Layer{ .conv1d = layer });
        } else if (LayerStruct == Conv2DLayer) {
            try self.layers.append(Layer{ .conv2d = layer });
        } else if (LayerStruct == Conv3DLayer) {
            try self.layers.append(Layer{ .conv3d = layer });
        } else if (LayerStruct == LSTMLayer) {
            try self.layers.append(Layer{ .lstm = layer });
        } else if (LayerStruct == LSTMCell) {
            try self.layers.append(Layer{ .lstm_cell = layer });
        } else if (LayerStruct == GRUCell) {
            try self.layers.append(Layer{ .gru_cell = layer });
        } else {
            @compileError("Unsupported layer type: " ++ @typeName(LayerStruct));
        }
    }