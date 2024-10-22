class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);



class NeuralNetwork {
  /*
   * If the first argument is a NeuralNetwork, the constructor clones it
   * USAGE: cloned_nn = new NeuralNetwork(to_clone_nn);
   */
  constructor(in_nodes, hidden_nodes_array, out_nodes) {
    if (in_nodes instanceof NeuralNetwork) {
      let a = in_nodes;

      this.input_nodes = a.input_nodes;
      this.hidden_nodes_array = a.hidden_nodes_array.slice(); // clone hidden layers array
      this.output_nodes = a.output_nodes;

      this.weights = a.weights.map(weight => weight.copy()); // clone weights for each layer
      this.biases = a.biases.map(bias => bias.copy()); // clone biases for each layer
    } else {
      this.input_nodes = in_nodes;
      this.hidden_nodes_array = hidden_nodes_array; // Array of hidden layer node counts
      this.output_nodes = out_nodes;

      // Initialize weights for each layer
      this.weights = [];
      
      this.weights.push(new Matrix(this.hidden_nodes_array[0], this.input_nodes)); // Input to first hidden layer
      for (let i = 0; i < this.hidden_nodes_array.length - 1; i++) {
        this.weights.push(new Matrix(this.hidden_nodes_array[i + 1], this.hidden_nodes_array[i])); // Between hidden layers
      }

      this.weights.push(new Matrix(this.output_nodes, this.hidden_nodes_array[this.hidden_nodes_array.length - 1])); // Last hidden to output
      this.weights.forEach(weight => weight.randomize());

      // Initialize biases for each layer
      this.biases = [];
      this.hidden_nodes_array.forEach(hidden_nodes => {
        this.biases.push(new Matrix(hidden_nodes, 1));
      });
      this.biases.push(new Matrix(this.output_nodes, 1)); // Output layer bias
      this.biases.forEach(bias => bias.randomize());
    }

    // TODO: copy these as well
    this.setLearningRate();
    this.setActivationFunction();
  }

  predict(input_array) {
    let inputs = Matrix.fromArray(input_array);

    // Generate hidden layer outputs step by step
    let current_output = inputs;
    for (let i = 0; i < this.weights.length - 1; i++) {
      let hidden = Matrix.multiply(this.weights[i], current_output);
      hidden.add(this.biases[i]);
      hidden.map(this.activation_function.func);
      current_output = hidden;
    }

    // Generate final output
    let output = Matrix.multiply(this.weights[this.weights.length - 1], current_output);
    output.add(this.biases[this.biases.length - 1]);
    output.map(this.activation_function.func);

    return output.toArray();
  }

  train(input_array, target_array) {
    // Generating the hidden outputs
    let inputs = Matrix.fromArray(input_array);
    let layer_outputs = [inputs]; // Store outputs of all layers for backpropagation

    // Forward pass
    let current_output = inputs;
    for (let i = 0; i < this.weights.length; i++) {
      let hidden = Matrix.multiply(this.weights[i], current_output);
      hidden.add(this.biases[i]);
      hidden.map(this.activation_function.func);
      layer_outputs.push(hidden);
      current_output = hidden;
    }

    // Backward pass
    let targets = Matrix.fromArray(target_array);
    let output_errors = Matrix.subtract(targets, layer_outputs[layer_outputs.length - 1]);

    for (let i = this.weights.length - 1; i >= 0; i--) {
      // Calculate gradients
      let gradients = Matrix.map(layer_outputs[i + 1], this.activation_function.dfunc);
      gradients.multiply(output_errors);
      gradients.multiply(this.learning_rate);

      // Calculate deltas
      let layer_T = Matrix.transpose(layer_outputs[i]);
      let weight_deltas = Matrix.multiply(gradients, layer_T);

      // Adjust weights and biases
      this.weights[i].add(weight_deltas);
      this.biases[i].add(gradients);

      // Calculate the errors for the next layer (backpropagation)
      if (i != 0) {
        let weights_T = Matrix.transpose(this.weights[i]);
        output_errors = Matrix.multiply(weights_T, output_errors);
      }
    }
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(func = sigmoid) {
    this.activation_function = func;
  }


  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
    nn.weights_ih = Matrix.deserialize(data.weights_ih);
    nn.weights_ho = Matrix.deserialize(data.weights_ho);
    nn.bias_h = Matrix.deserialize(data.bias_h);
    nn.bias_o = Matrix.deserialize(data.bias_o);
    nn.learning_rate = data.learning_rate;
    return nn;
  }

}
