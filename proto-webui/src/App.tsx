import * as React from 'react';

import { Button, Container, Grid, Icon, Menu } from 'semantic-ui-react';

import About from './components/About';


import 'semantic-ui-css/semantic.min.css';
import './App.css';

import * as GPU from 'gpu.js';

const gpu = new GPU();

const RES_X = 4192;
const RES_Y = 4192;
const N_GAUSS = 7;

function randMean() {
  return (Math.random() * 0.8) + 0.1
}

const MEANS = [
  [randMean(), randMean()],
  [randMean(), randMean()],
  [randMean(), randMean()],
  [randMean(), randMean()],
  [randMean(), randMean()],
  [randMean(), randMean()],
  [randMean(), randMean()]
];

function randStd() {
  return (Math.random() * 0.16) + 0.04;
}

const STDS = [
  randStd(),
  randStd(),
  randStd(),
  randStd(),
  randStd(),
  randStd(),
  randStd()
]

const gaussianPDFKernel = gpu.createKernel(function(this: any, ngauss: number, means : number[][], stds : number[], bounds : number[]) {
  let ov = 0;
  for (let i = 0; i < ngauss; i++) {
    const variance = stds[i] * stds[i];
    const x = this.thread.x / bounds[0];
    const y = this.thread.y / bounds[1];
    const dx = means[i][0] - x;
    const dy = means[i][1] - y;
    const pd1 = 1.0 / Math.sqrt(2 * 3.141593 * variance)
    ov += pd1 * Math.exp(-((dx * dx) + (dy * dy)) / (2 * variance));
  }
  return ov;
}).setOutput([RES_X, RES_Y])
  .setOutputToTexture(true);

const computeFinalValuesKernel = gpu.createKernel(function(this: any, scale : number[], max : number[], pdfs: number[][]) {
  const ov = pdfs[this.thread.x][this.thread.y];
  /* for (let i = 0; i < ngauss; i++) {
    ov += pdfs[i][this.thread.y][this.thread.x]
  }*/
  const x = this.thread.x / max[0];
  const y = this.thread.y / max[1];
  return (x * scale[0]) + (y * scale[1]) + ov;
}).setOutput([RES_X, RES_Y])
  .setOutputToTexture(true);

const colorMapKernel = gpu.createKernel(function(this: any, inVals : number[][], vMax : number) {
  const scale = (inVals[this.thread.x][this.thread.y] / vMax);
  let red = 255;
  let blue = 0;
  let green = 0;
  
  if (scale < 0.25) {
    red = scale / 0.25;
  }

  if (scale > 0.50 && scale < 0.75) {
    blue = (scale-0.5) / 0.25;
  } else if (scale > 0.75) {
    blue = 255;
  }

  if (scale > 0.75) {
    green = (scale - 0.75) / 0.25
  }

  this.color(red, blue, green);
}).setOutput([RES_X, RES_Y])
  .setGraphical(true);

interface IMainState {
  getCanvas: () => HTMLCanvasElement
}

class App extends React.Component<{}, IMainState> {

  constructor(props : {}) {
    super(props);
    this.state = {
      getCanvas() {
        console.time('renderCanvas');
        const tx1 = gaussianPDFKernel(N_GAUSS, MEANS, STDS, [RES_X, RES_Y]);
        const tx2 = computeFinalValuesKernel([Math.random()*2.5 + 3, Math.random()*2.5 + 3], [RES_X, RES_Y], tx1)
        colorMapKernel(tx2, 16);
        const c = colorMapKernel.getCanvas();
        console.timeEnd('renderCanvas');
        return c;
      }
    }

    // this.state.kernel();
  }

  public componentDidMount() {
    const container = document.getElementById('canvasContainer');
    if (container != null) {
      container.appendChild(this.state.getCanvas());
    }
  }

  public render() {
    return (
      <div className="App">
        <Menu size="huge" fixed="top" inverted={true}>
          <Menu.Item><Icon name="area chart" />Bayesian Optimization</Menu.Item>
          <Menu.Item position="right">
            <About />
          </Menu.Item>
          <Menu.Item  link={true}>
            <a href="https://github.com/willmtemple"><Icon name="github" />On GitHub</a>
          </Menu.Item>
        </Menu>
        <div className='appBody'>
          <Grid stretched={true} stackable={true} divided={true} padded="horizontally">
            <Grid.Row>
              <Grid.Column width={8}>
                <div className="cancon" id="canvasContainer" />
              </Grid.Column>
              <Grid.Column width={8}>
                <div className="cancon" id="canvasContainer2" />
              </Grid.Column>
            </Grid.Row>
            <Grid.Row>
              <Grid.Column width={8}>
                <div className="cancon" id="canvasContainer3" />
              </Grid.Column>
              <Grid.Column width={8}>
                <div className="cancon" id="canvasContainer4" />
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </div>
        <Menu size="massive" fixed="bottom">
          <Container>
            <Menu.Item position="left">
              <Button color="red">
                <Icon name="refresh" /> Initialize
              </Button>
            </Menu.Item>
            <Menu.Item>
              <Button disabled={true}>
                1. Find New Point
              </Button>
            </Menu.Item>
            <Menu.Item>
              <Button disabled={true}>
                2. Sample Point
              </Button>
            </Menu.Item>
            <Menu.Item>
              <Button disabled={true}>
                3. Re-Fit Model
              </Button>
            </Menu.Item>
            <Menu.Item position="right">
              <Button disabled={true} color="green">
                <Icon name="play" /> Autorun
              </Button>
            </Menu.Item>
          </Container>
        </Menu>
      </div>
    );
  }
}

export default App;
