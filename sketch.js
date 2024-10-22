let nn;
let lr_slider;

let training_data = [
  {
    inputs: [0, 0],
    output: [0],
  },
  {
    inputs: [0, 1],
    output: [1],
  },
  {
    inputs: [1, 0],
    output: [1],
  },
  {
    inputs: [1, 1],
    output: [0],
  },
];

function setup() {
  createCanvas(400, 400);
  nn = new NeuralNetwork(2, [100], 1);
  lr_slider=createSlider(0.01, 0.5,0.1,0.01);
}

function draw() {
  background(0);

for(let i=0;i<1000;i++){
    let data=random(training_data);
    nn.train(data.inputs,data.output);
}  

nn.setLearningRate(lr_slider.value());
 
let resoluction=10;
let cols=width/resoluction;
let rows=height/resoluction;

for(let i=0;i<cols;i++){
    for(let j=0;j<cols;j++){
        let x1=i/cols;
        let x2=j/rows;
        let inputs=[x1,x2];
       // noStroke();
        let y=nn.predict(inputs)
        fill(y*255);
        rect(i*resoluction,j*resoluction,resoluction,resoluction)
    }
}

}
