let nn;

let one_data,
  two_data,
  three_data,
  four_data,
  five_data,
  six_data,
  seven_data,
  eight_data,
  nine_data;
let cats_training, trains_training, rainbows_training;
let cats_testing, trains_testing, rainbows_testing;
let json;

let cats = {};
let trains = {};
let rainbows = {};

const ZERO = 0;
const ONE = 1;
const TWO = 2;
const THREE = 3;
const FOUR = 4;
const FIVE = 5;
const SIX = 6;
const SEVEN = 7;
const EIGHT = 8;
const NINE = 9;

const len = 784;
const total_data = 100000;

function preload() {
  cats_data = loadBytes("./data/cat1000.bin");
  trains_data = loadBytes("./data/train1000.bin");
  rainbows_data = loadBytes("./data/rainbow1000.bin");
}

async function loadNeuralNetwork(path) {
  return new Promise((resolve, reject) => {
    loadJSON(
      path,
      (jsonData) => resolve(NeuralNetwork.deserialize(jsonData)),
      (error) => reject(error)
    );
  });
}

function prepareData(category, data, label) {
  category.training = [];
  category.testing = [];

  for (let i = 0; i < total_data; i++) {
    let threshold = floor(0.8 * total_data);
    if (i < threshold) {
      let offset = i * len;
      category.training[i] = data.bytes.subarray(offset, offset + len);
      category.training[i].label = label;
    } else {
      let offset = i * len;
      category.testing[i - threshold] = data.bytes.subarray(
        offset,
        offset + len
      );
      category.testing[i - threshold].label = label;
    }
  }
}

async function setup() {
  createCanvas(280, 280);
  background(255);

  let epochCounter = 0;

  prepareData(cats, cats_data, CAT);
  prepareData(trains, trains_data, TRAIN);
  prepareData(rainbows, rainbows_data, RAINBOW);

  // nn = await loadNeuralNetwork('./data/trained.json');
  nn = new NeuralNetwork(len, [64], 10);
  nn.learningRate = 0.01;

  let training = [].concat(cats.training, rainbows.training, trains.training);
  let testing = [].concat(cats.testing, rainbows.testing, trains.testing);

  let trainButton = select("#train");
  trainButton.mousePressed(() => {
    for (let i = 0; i < 1; i++) {
      epochCounter++;
      trainEpoch(training);
      console.log(`${epochCounter} Epoch conpleted`);
    }
  });

  let testButton = select("#test");
  testButton.mousePressed(() => {
    let accuracy = testAll(testing);
    console.log(`Final accuracy: ${nfc(accuracy, 2)}%`);
  });

  let clearButton = select("#clear");
  clearButton.mousePressed(() => {
    background(255);
  });

  let guessButton = select("#guess");
  guessButton.mousePressed(() => {
    let inputs = [];
    let img = get();
    img.resize(28, 28);

    img.loadPixels();

    for (let i = 0; i < len; i++) {
      let brightness = img.pixels[i * 4];
      inputs[i] = (255 - brightness) / 255.0;
    }

    console.log(inputs);

    let guess = nn.predict(inputs);
    let m = max(guess);
    let classification = guess.indexOf(m);

    switch (classification) {
      case ZERO:
        console.log("Zero");
        break;
      case ONE:
        console.log("One");
        break;
      case TWO:
        console.log("Two");
        break;
      case THREE:
        console.log("Three");
        break;
      case FOUR:
        console.log("Four");
        break;
      case FIVE:
        console.log("Five");
        break;
      case SIX:
        console.log("Six");
        break;
      case SEVEN:
        console.log("Seven");
        break;
      case EIGHT:
        console.log("Eight");
        break;
      case NINE:
        console.log("Nin");
        break;
    }
  });
}

function trainEpoch(training) {
  shuffle(training, true);
  for (let i = 0; i < training.length; i++) {
    let inputs = [];

    let data = training[i];

    for (let j = 0; j < data.length; j++) {
      inputs[j] = data[j] / 255.0;
    }
    let label = training[i].label;
    let targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    targets[label] = 1;
    nn.train(inputs, targets);
    console.log("training....");
  }
}

function testAll(testing) {
  let correct = 0;
  for (let i = 0; i < testing.length; i++) {
    let inputs = [];
    let data = testing[i];

    for (let j = 0; j < data.length; j++) {
      inputs[j] = data[j] / 255.0;
    }

    let label = testing[i].label;
    let guess = nn.predict(inputs);

    let m = max(guess);
    let classification = guess.indexOf(m);

    if (classification === label) {
      correct++;
    }
  }
  let presentage = (correct / testing.length) * 100;
  return presentage;
}

function draw() {
  strokeWeight(10);
  stroke(0);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}
