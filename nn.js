class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  (x) => 1 / (1 + Math.exp(-x)),
  (y) => y * (1 - y)
);

let tanh = new ActivationFunction(
  (x) => Math.tanh(x),
  (y) => 1 - y * y
);



class NeuralNetwork {
 

  constructor(in_nodes, hidden_nodes_array, out_nodes) {
    if (in_nodes instanceof NeuralNetwork) {
      let a = in_nodes;

      this.input_nodes = a.input_nodes;
      this.hidden_nodes_array = a.hidden_nodes_array.slice(); 
      this.output_nodes = a.output_nodes;

      this.weights = a.weights.map((weight) => weight.copy()); 
      this.biases = a.biases.map((bias) => bias.copy()); 
    } else {
      this.input_nodes = in_nodes;
      this.hidden_nodes_array = hidden_nodes_array; 
      this.output_nodes = out_nodes;

      
      this.weights = [];

      this.weights.push(
        new Matrix(this.hidden_nodes_array[0], this.input_nodes)
      ); 
      for (let i = 0; i < this.hidden_nodes_array.length - 1; i++) {
        this.weights.push(
          new Matrix(this.hidden_nodes_array[i + 1], this.hidden_nodes_array[i])
        ); 
      }

      this.weights.push(
        new Matrix(
          this.output_nodes,
          this.hidden_nodes_array[this.hidden_nodes_array.length - 1]
        )
      ); 
      this.weights.forEach((weight) => weight.randomize());

      this.biases = [];
      this.hidden_nodes_array.forEach((hidden_nodes) => {
        this.biases.push(new Matrix(hidden_nodes, 1));
      });
      this.biases.push(new Matrix(this.output_nodes, 1)); 
      this.biases.forEach((bias) => bias.randomize());
    }

  
    this.setLearningRate();
    this.setActivationFunction();
  }

  predict(input_array) {
    let inputs = Matrix.fromArray(input_array);

    let current_output = inputs;
    for (let i = 0; i < this.weights.length - 1; i++) {
      let hidden = Matrix.multiply(this.weights[i], current_output);
      hidden.add(this.biases[i]);
      hidden.map(this.activation_function.func);
      current_output = hidden;
    }

    let output = Matrix.multiply(
      this.weights[this.weights.length - 1],
      current_output
    );
    output.add(this.biases[this.biases.length - 1]);

    output.map(this.activation_function.func);

    return output.toArray();
  }

  train(input_array, target_array) {
    let inputs = Matrix.fromArray(input_array);
    let layer_outputs = [inputs]; 

    let current_output = inputs;
    for (let i = 0; i < this.weights.length; i++) {
      let hidden = Matrix.multiply(this.weights[i], current_output);
      hidden.add(this.biases[i]);
      hidden.map(this.activation_function.func);
      layer_outputs.push(hidden);
      current_output = hidden;
    }

    let targets = Matrix.fromArray(target_array);
    let output_errors = Matrix.subtract(
      targets,
      layer_outputs[layer_outputs.length - 1]
    );

    for (let i = this.weights.length - 1; i >= 0; i--) {

      let gradients = Matrix.map(
        layer_outputs[i + 1],
        this.activation_function.dfunc
      );
      gradients.multiply(output_errors);
      gradients.multiply(this.learning_rate);


      let layer_T = Matrix.transpose(layer_outputs[i]);
      let weight_deltas = Matrix.multiply(gradients, layer_T);


      this.weights[i].add(weight_deltas);
      this.biases[i].add(gradients);


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
    return JSON.stringify({
      input_nodes: this.input_nodes,
      hidden_nodes_array: this.hidden_nodes_array,
      output_nodes: this.output_nodes,
      weights: this.weights.map(weight => weight.serialize()),
      biases: this.biases.map(bias => bias.serialize()),
      learning_rate: this.learning_rate
    });
  }
  

  static deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes_array, data.output_nodes);
    nn.weights = data.weights.map(Matrix.deserialize);
    nn.biases = data.biases.map(Matrix.deserialize);
    nn.learning_rate = data.learning_rate;
  
    return nn;
  }
  
}
