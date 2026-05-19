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



The parent layer struct should contain the common fields and methods for all layers.

every layer is a child of the parent layer strcut
parent layer strcut incldues a layer type feild(this feild points to the child object struct)
when defining a child the layer. The type feild is set by defualt to the child type(what that layer is inteded to be)
eg layer.linear.type points to the linear layer object

Using this revamp this code and any other relevant code to implement the new layer system


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

now shoudl fucntion as so 

        ///TODO check isinstance(layer) else raise typer error
        /// check self.layers.append(Layer{layer.type});