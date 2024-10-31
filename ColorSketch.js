let brain;
let r, g, b;

let which = "black";

function pickColor() {
  r = random(255);
  g = random(255);
  b = random(255);
  redraw();
}

function coloPredictor(r, g, b) {
  let inputs = [r / 255, g / 255, b / 255];
  let outputs = brain.predict(inputs);

  return outputs[0] > outputs[1] ? "black" : "white";
}

function setup() {
  brain = new NeuralNetwork(3, [30], 2);

  createCanvas(600, 300);
  noLoop();

  for (let i = 0; i < 1000; i++) {
    let r = random(255);
    let g = random(255);
    let b = random(255);

    let inputs = [r / 255, g / 255, b / 255];
    let targets = trainColor(r, g, b);

    brain.train(inputs, targets);
  }

  pickColor();
}

function trainColor(r, g, b) {
  return r + g + b > 300 ? [1, 0] : [0, 1];
}

function mousePressed() {
//   let targets = mouseX > width / 2 ? [0, 1] : [1, 0];
//   let inputs = [r / 255, g / 255, b / 255];
//   console.log(targets);

//   brain.train(inputs, targets);

  pickColor();
}

function draw() {
  background(r, g, b);

  strokeWeight(4);
  stroke(0);
  line(width / 2, 0, width / 2, height);

  textSize(64);
  noStroke();
  fill(0);
  textAlign(CENTER, CENTER);
  text("black", 150, 150);
  fill(255);
  text("white", 450, 150);

  which = coloPredictor(r, g, b);
  fill(which === "black" ? 0 : 255);
  ellipse(which === "black" ? 150 : 450, 230, 60, 60);
}
