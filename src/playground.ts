/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem,
  RegressionDatasetKind
} from "./state";
import {
  Example2D,
  RandomWalkRegion,
  randomWalkRegressionData,
  shuffle
} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

declare var require: any;
let THREE: any = require("three");
let mainWidth;

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

let inputLabels: {[name: string]: string} = {};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Random walk dimension", "dimensionN"],
  ["Random walk length", "walkLengthK"],
  ["Sample multiplier", "sampleMultiplierM"],
  ["Gaussian noise controls", "noiseGaussian"],
  ["Architecture frontend", "architectureFrontend"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();
if (state.useTransformer && state.useCnnFrontend) {
  state.useCnnFrontend = false;
}

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
d3.select("#heatmap").select("div:last-child").classed("heatmap-2d", true);
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let randomWalkRegions: RandomWalkRegion[] = [];
let randomWalkPath: number[][] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);
interface ProjectionViewState {
  scene: any;
  camera: any;
  renderer: any;
  root: any;
  points: any;
  dynamicObjects: any[];
  webglFailed: boolean;
  controlsAttached: boolean;
}
interface OodPlanePayload {
  positions: number[];
  colors: number[];
  indices: number[];
  maxError: number;
}
interface ProjectionPayload {
  positions: number[];
  colors: number[];
  mode: string;
  range: number;
  oodPlane: OodPlanePayload;
}
interface Heatmap3DPayload {
  truthLinePositions: number[];
  truthLineColors: number[];
  modelLinePositions: number[];
  modelLineColors: number[];
  oodPlane: OodPlanePayload;
  mode: string;
  axes: string;
  range: number;
  lineScale: number;
  maxError: number;
}
let projectionViews: {[name: string]: ProjectionViewState} = {};
const MAX_RANDOM_WALK_REGIONS = 10000;
const OOD_PLANE_RESOLUTION = 22;

function usingRandomWalk(): boolean {
  return state.problem === Problem.REGRESSION &&
      state.regressionDatasetKind === RegressionDatasetKind.RANDOM_WALK;
}

function getOutputDimension(): number {
  return usingRandomWalk() ? Math.max(1, Math.floor(state.dimensionN)) : 1;
}

function getFrontendOutputSize(rawInputSize: number): number {
  if (state.useTransformer) {
    return 16;
  }
  if (state.useCnnFrontend) {
    return 8;
  }
  return rawInputSize;
}

function getTarget(point: Example2D): number | number[] {
  return point.labelVec != null ? point.labelVec : point.label;
}

function getMaxRandomWalkK(dimension = state.dimensionN): number {
  let safeDimension = Math.max(1, Math.floor(dimension));
  return Math.max(2, Math.floor(Math.pow(MAX_RANDOM_WALK_REGIONS,
      1 / safeDimension)));
}

function clampRandomWalkK(value: number, dimension = state.dimensionN): number {
  return Math.max(2, Math.min(getMaxRandomWalkK(dimension),
      Math.floor(value)));
}

function normalizeRandomWalkControls() {
  state.dimensionN = Math.max(1, Math.min(8, Math.floor(state.dimensionN)));
  state.walkLengthK = clampRandomWalkK(state.walkLengthK);
  let maxK = getMaxRandomWalkK();
  d3.select("#walkLengthK").attr("max", maxK);
  d3.select("label[for='walkLengthK'] .value").text(state.walkLengthK);
}

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    state.regressionDatasetKind = RegressionDatasetKind.LEGACY;
    regDataThumbnails.classed("selected", false);
    d3.select("#random-walk-dataset").classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", state.regressionDatasetKind !==
        RegressionDatasetKind.RANDOM_WALK);

  d3.select("#random-walk-dataset").on("click", function() {
    if (state.regressionDatasetKind === RegressionDatasetKind.RANDOM_WALK) {
      return;
    }
    state.regressionDatasetKind = RegressionDatasetKind.RANDOM_WALK;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  }).classed("selected", state.regressionDatasetKind ===
      RegressionDatasetKind.RANDOM_WALK);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let dimensionN = d3.select("#dimensionN").on("input", function() {
    state.dimensionN = Math.max(1, Math.min(8, Math.floor(+this.value)));
    state.walkLengthK = clampRandomWalkK(state.walkLengthK);
    d3.select("#walkLengthK")
        .attr("max", getMaxRandomWalkK())
        .property("value", state.walkLengthK);
    d3.select("label[for='dimensionN'] .value").text(state.dimensionN);
    d3.select("label[for='walkLengthK'] .value").text(state.walkLengthK);
    generateData(true);
    parametersChanged = true;
    reset();
  });
  normalizeRandomWalkControls();
  dimensionN.property("value", state.dimensionN);
  d3.select("label[for='dimensionN'] .value").text(state.dimensionN);

  let walkLengthK = d3.select("#walkLengthK").on("input", function() {
    state.walkLengthK = clampRandomWalkK(+this.value);
    d3.select(this).property("value", state.walkLengthK);
    d3.select("label[for='walkLengthK'] .value").text(state.walkLengthK);
    generateData(true);
    parametersChanged = true;
    reset();
  });
  walkLengthK.property("value", state.walkLengthK);
  d3.select("label[for='walkLengthK'] .value").text(state.walkLengthK);

  let sampleMultiplierM = d3.select("#sampleMultiplierM").on("input",
      function() {
    state.sampleMultiplierM = Math.max(1, Math.floor(+this.value));
    d3.select("label[for='sampleMultiplierM'] .value")
        .text(state.sampleMultiplierM);
    generateData(true);
    parametersChanged = true;
    reset();
  });
  sampleMultiplierM.property("value", state.sampleMultiplierM);
  d3.select("label[for='sampleMultiplierM'] .value")
      .text(state.sampleMultiplierM);

  let seedInput = d3.select("#seed").on("change", function() {
    if (this.value != null && this.value !== "") {
      state.seed = this.value;
      generateData(true);
      parametersChanged = true;
      reset();
    }
  });
  seedInput.property("value", state.seed);

  let noiseEnabled = d3.select("#noiseEnabled").on("change", function() {
    state.noiseEnabled = this.checked;
    generateData(true);
    parametersChanged = true;
    reset();
  });
  noiseEnabled.property("checked", state.noiseEnabled);

  let noiseMean = d3.select("#noiseMean").on("change", function() {
    state.noiseMean = +this.value;
    generateData(true);
    parametersChanged = true;
    reset();
  });
  noiseMean.property("value", state.noiseMean);

  let noiseVariance = d3.select("#noiseVariance").on("change", function() {
    state.noiseVariance = Math.max(0, +this.value);
    generateData(true);
    parametersChanged = true;
    reset();
  });
  noiseVariance.property("value", state.noiseVariance);

  let useCnnFrontend = d3.select("#useCnnFrontend").on("change", function() {
    state.useCnnFrontend = this.checked;
    if (state.useCnnFrontend) {
      state.useTransformer = false;
      d3.select("#useTransformer").property("checked", false);
    }
    parametersChanged = true;
    reset();
  });
  useCnnFrontend.property("checked", state.useCnnFrontend);

  let useTransformer = d3.select("#useTransformer").on("change", function() {
    state.useTransformer = this.checked;
    if (state.useTransformer) {
      state.useCnnFrontend = false;
      d3.select("#useCnnFrontend").property("checked", false);
    }
    parametersChanged = true;
    reset();
  });
  useTransformer.property("checked", state.useTransformer);

  let deadNeuronEps = d3.select("#deadNeuronEps").on("change", function() {
    state.deadNeuronEps = Math.max(0, +this.value);
    state.serialize();
    updateDeadNeuronMonitor();
  });
  deadNeuronEps.property("value", state.deadNeuronEps);

  let sweepEpochs = d3.select("#sweepEpochs").on("change", function() {
    state.sweepEpochs = Math.max(1, Math.floor(+this.value));
    state.serialize();
  });
  sweepEpochs.property("value", state.sweepEpochs);

  d3.select("#sweep-depth-button").on("click", function() {
    runAutoSweep("depth");
  });
  d3.select("#sweep-width-button").on("click", function() {
    runAutoSweep("width");
  });

  ["projectionX", "projectionY", "projectionZ"].forEach(prop => {
    d3.select("#" + prop).on("change", function() {
      state[prop] = +this.value;
      updateProjectionControls();
      state.serialize();
      drawRandomWalkProjection();
    });
  });

  ["projectionRotateX", "projectionRotateY", "projectionRotateZ",
      "projectionScale", "projectionRange"].forEach(prop => {
    let control = d3.select("#" + prop).on("input", function() {
      state[prop] = +this.value;
      if (prop === "projectionRange") {
        state.projectionRange = Math.max(0.1, state.projectionRange);
        d3.select("#projectionRangeValue").text(state.projectionRange);
      }
      state.serialize();
      drawRandomWalkProjection();
    });
    control.property("value", state[prop]);
  });
  d3.select("#projectionRangeValue").text(state.projectionRange);

  let showOodPlane = d3.select("#showOodPlane").on("change", function() {
    state.showOodPlane = this.checked;
    state.serialize();
    drawRandomWalkProjection();
  });
  showOodPlane.property("checked", state.showOodPlane);

  d3.select("#oodPlaneAxis").on("change", function() {
    state.oodPlaneAxis = Math.max(0, Math.min(2, +this.value));
    state.serialize();
    drawRandomWalkProjection();
  });

  let oodPlaneOffset = d3.select("#oodPlaneOffset").on("input", function() {
    state.oodPlaneOffset = Math.max(-1, Math.min(1, +this.value));
    d3.select("#oodPlaneOffsetValue").text(state.oodPlaneOffset.toFixed(2));
    state.serialize();
    drawRandomWalkProjection();
  });
  oodPlaneOffset.property("value", state.oodPlaneOffset);
  d3.select("#oodPlaneOffsetValue").text(state.oodPlaneOffset.toFixed(2));

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let inputFeature = INPUTS[nodeId];
    let label = inputLabels[nodeId] ||
        (inputFeature != null && inputFeature.label != null ?
        inputFeature.label : nodeId);
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(inputFeature == null ? "active" : activeOrNotClass,
        true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      drawRandomWalkProjection();
      if (!usingRandomWalk()) {
        updateDecisionBoundary(network, false);
        heatMap.updateBackground(boundary[nodeId], state.discretize);
      }
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      drawRandomWalkProjection();
      if (!usingRandomWalk()) {
        updateDecisionBoundary(network, false);
        heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
            state.discretize);
      }
    });
  if (isInput) {
    div.on("click", function() {
      if (!(nodeId in INPUTS)) {
        return;
      }
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    if (nodeId in INPUTS) {
      div.style("cursor", "pointer");
    }
  }
  if (isInput) {
    div.classed(nodeId in INPUTS ? activeOrNotClass : "active", true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = network[0].map(node => node.id);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output layer separately.
  cx = width + RECT_SIZE / 2;
  let outputLayer = network[numLayers - 1];
  maxY = Math.max(maxY, nodeIndexScale(outputLayer.length));
  for (let outputIndex = 0; outputIndex < outputLayer.length; outputIndex++) {
    let node = outputLayer[outputIndex];
    let cy = nodeIndexScale(outputIndex) + RECT_SIZE / 2;
    node2coord[node.id] = {cx, cy};
    for (let i = 0; i < node.inputLinks.length; i++) {
      let link = node.inputLinks[i];
      drawLink(link, node2coord, network, container, i === 0, i,
          node.inputLinks.length);
    }
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    network[0].forEach(node => {
      boundary[node.id] = new Array(DENSITY);
    });
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      network[0].forEach(node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        network[0].forEach((node, inputIndex) => {
          boundary[node.id][i][j] = input[inputIndex];
        });
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint);
    let output = nn.forwardPropOutputs(network, input);
    let target = getTarget(dataPoint);
    let targets = Array.isArray(target) ? target : [target];
    for (let outputIndex = 0; outputIndex < output.length; outputIndex++) {
      loss += nn.Errors.SQUARE.error(output[outputIndex],
          targets[outputIndex]);
    }
  }
  return loss / (dataPoints.length * getOutputDimension());
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  if (usingRandomWalk()) {
    drawRandomWalkProjection();
    updateLossAndIterationUI();
    return;
  }
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });
  drawRandomWalkProjection();

  updateLossAndIterationUI();
}

function updateLossAndIterationUI() {
  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    if (!isFinite(n)) {
      return String(n);
    }
    let abs = Math.abs(n);
    if (abs > 0 && abs < 0.001) {
      return n.toExponential(2);
    }
    if (abs < 1) {
      return n.toFixed(5);
    }
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
  updateDeadNeuronMonitor();
}

function updateProjectionControls() {
  ["projectionX", "projectionY", "projectionZ"].forEach((prop, i) => {
    let select = d3.select("#" + prop);
    if (!select.size()) {
      return;
    }
    let options = select.selectAll("option").data(d3.range(getOutputDimension()));
    options.enter().append("option");
    options.attr("value", (d: number) => d).text((d: number) => "X" + (d + 1));
    options.exit().remove();
    if (state[prop] == null || state[prop] >= getOutputDimension()) {
      state[prop] = Math.min(i, getOutputDimension() - 1);
    }
    select.property("value", state[prop]);
  });
  let planeAxis = d3.select("#oodPlaneAxis");
  if (planeAxis.size()) {
    let labels = getProjectionAxisLabels();
    let options = planeAxis.selectAll("option").data([0, 1, 2]);
    options.enter().append("option");
    options.attr("value", (d: number) => d)
        .text((d: number) => labels[d] + " fixed");
    options.exit().remove();
    state.oodPlaneAxis = Math.max(0, Math.min(2,
        Math.floor(state.oodPlaneAxis)));
    planeAxis.property("value", state.oodPlaneAxis);
  }
}

function drawRandomWalkProjection() {
  updateOutputProjectionVisibility();
  d3.select(".random-projection").style("display",
      usingRandomWalk() ? null : "none");
  if (!usingRandomWalk()) {
    return;
  }
  let largeContainer = document.querySelector("#random-walk-projection") as
      HTMLDivElement;
  let outputContainer = document.querySelector("#output-3d-projection") as
      HTMLDivElement;
  drawOutputHeatmap3D("large", largeContainer, "#projection-mode", true);
  drawOutputHeatmap3D("output", outputContainer, "#output-3d-mode", false);
}

function updateOutputProjectionVisibility() {
  let show3d = usingRandomWalk();
  d3.select("#heatmap").classed("showing-3d-output", show3d);
  d3.select("#output-3d-projection")
      .style("display", show3d ? "block" : "none");
  d3.select("#output-3d-mode")
      .style("display", show3d ? "block" : "none");
  d3.select("#heatmap").selectAll(".heatmap-2d")
      .style("display", show3d ? "none" : null);
}

function drawProjectionView(name: string, containerSelector: string,
    modeSelector: string, useNetworkOutput: boolean) {
  let container = document.querySelector(containerSelector) as HTMLDivElement;
  if (container == null) {
    return;
  }
  if (name === "output") {
    drawOutputHeatmap3D(name, container, modeSelector);
    return;
  }
  let payload = buildRandomWalkProjectionPayload(useNetworkOutput);
  d3.select(modeSelector).text("Color: " + payload.mode +
      " | axes: " + getProjectionAxisLabels().join("/") +
      " | OOD plane: " + (payload.oodPlane == null ? "off" :
          ("MSE max " + payload.oodPlane.maxError.toFixed(3))) +
      " | View: [-" + payload.range + ", " + payload.range + "]");
  ensureProjectionScene(name, container);
  let view = projectionViews[name];
  if (view == null || view.webglFailed) {
    drawFallbackProjection(container, payload.positions, payload.colors);
    return;
  }
  clearProjectionDynamicObjects(view);
  if (payload.oodPlane != null) {
    addProjectionObject(view, createColoredSurface(payload.oodPlane.positions,
        payload.oodPlane.colors, payload.oodPlane.indices, 0.38));
  }
  view.points = createPointCloud(payload.positions, payload.colors, 0.045,
      0.88);
  addProjectionObject(view, view.points);
  renderProjectionView(name);
}

function drawOutputHeatmap3D(name: string, container: HTMLDivElement,
    modeSelector: string, showPlane = false) {
  if (container == null) {
    return;
  }
  let payload = buildOutputHeatmap3DPayload(showPlane);
  d3.select(modeSelector).text("Output space | Truth line: random walk | " +
      payload.mode +
      " | axes: " + payload.axes +
      " | line scale x" + payload.lineScale.toFixed(1) +
      (payload.oodPlane == null ? "" :
          (" | OOD plane MSE max " + payload.oodPlane.maxError.toFixed(3))) +
      " | MSE max " + payload.maxError.toFixed(3) +
      " | View: [-" + payload.range + ", " + payload.range + "]");
  ensureProjectionScene(name, container);
  let view = projectionViews[name];
  if (view == null || view.webglFailed) {
    drawFallbackProjection(container, payload.truthLinePositions,
        payload.truthLineColors);
    return;
  }
  clearProjectionDynamicObjects(view);
  let largeView = name === "large";
  if (payload.oodPlane != null) {
    addProjectionObject(view, createColoredSurface(payload.oodPlane.positions,
        payload.oodPlane.colors, payload.oodPlane.indices, 0.42));
  }
  addProjectionObject(view, createSegmentedLine(payload.truthLinePositions,
      payload.truthLineColors, largeView ? 0.018 : 0.013, 0.96));
  if (payload.modelLinePositions.length >= 6) {
    addProjectionObject(view, createSegmentedLine(payload.modelLinePositions,
        payload.modelLineColors, largeView ? 0.014 : 0.009, 0.82));
  }
  renderProjectionView(name);
}

function buildRandomWalkProjectionPayload(useNetworkOutput: boolean):
    ProjectionPayload {
  let px = Math.min(state.projectionX, getOutputDimension() - 1);
  let py = Math.min(state.projectionY, getOutputDimension() - 1);
  let pz = Math.min(state.projectionZ, getOutputDimension() - 1);
  let range = getProjectionRange();
  let allData = trainData.concat(testData);
  let positions: number[] = [];
  let colors: number[] = [];
  let mode = selectedNodeId != null ? "node " + selectedNodeId :
      (useNetworkOutput ? "model y1" : "target y1");
  allData.forEach(point => {
    let input = point.inputs || [point.x, point.y, 0];
    positions.push(normalizeProjectionValue(input[px] || 0, range),
        normalizeProjectionValue(input[py] || 0, range),
        normalizeProjectionValue(input[pz] || 0, range));
    let value = getProjectionColorValue(point, useNetworkOutput);
    let color = d3.rgb(colorScale(value) as any);
    colors.push(color.r / 255, color.g / 255, color.b / 255);
  });
  return {positions, colors, mode, range, oodPlane: null};
}

function buildOutputHeatmap3DPayload(includePlane = false): Heatmap3DPayload {
  let axes = getProjectionAxes();
  let range = getProjectionRange();
  let orderedRegions = getRegionsByWalkStep();
  let truthVectors = randomWalkPath.length > 0 ? randomWalkPath :
      orderedRegions.map(region => region.targetVec);
  if (truthVectors.length > 0) {
    truthVectors = [zeroVector(getOutputDimension())].concat(truthVectors);
  }
  let truthLinePositions: number[] = [];
  let truthValues: number[] = [];
  truthVectors.forEach(vector => {
    addPosition(truthLinePositions, vector, axes, range);
    truthValues.push(vector[0] || 0);
  });
  let modelLinePositions: number[] = [];
  let modelValues: number[] = [];
  let modelVectors: number[][] = [];
  let mode = selectedNodeId != null ? "node color: " + selectedNodeId :
      "model line color: MSE";
  orderedRegions.forEach(region => {
    let point = exampleFromRegion(region);
    if (selectedNodeId == null) {
      let output = getModelOutputVector(point);
      modelVectors.push(output.slice());
      addPosition(modelLinePositions, output, axes, range);
      modelValues.push(getVectorMse(output, region.targetVec));
    } else {
      addPosition(modelLinePositions, region.targetVec, axes, range);
      modelValues.push(getNodeProjectionValue(point, selectedNodeId));
    }
  });
  let errorScale = selectedNodeId == null ?
      makeMseColorScale(modelValues) :
      makeRandomWalkValueColorScale(truthValues.concat(modelValues));
  let lineScale = getOutputLineDisplayScale(truthLinePositions.concat(
      modelLinePositions));
  let oodPlane = includePlane && selectedNodeId == null ?
      buildOutputSpaceOodPlanePayload(range, modelVectors) : null;
  scalePositions(truthLinePositions, lineScale);
  scalePositions(modelLinePositions, lineScale);
  if (oodPlane != null) {
    scalePositions(oodPlane.positions, lineScale);
  }
  return {
    truthLinePositions,
    truthLineColors: solidColors(truthVectors.length, "#222222"),
    modelLinePositions,
    modelLineColors: valuesToColors(modelValues, errorScale),
    oodPlane,
    mode,
    axes: getProjectionAxisLabels().join("/"),
    range,
    lineScale,
    maxError: maxFinite(modelValues)
  };
}

function zeroVector(length: number): number[] {
  let result = new Array(length);
  for (let i = 0; i < length; i++) {
    result[i] = 0;
  }
  return result;
}

function getOutputLineDisplayScale(positions: number[]): number {
  let maxAbs = 0;
  positions.forEach(value => maxAbs = Math.max(maxAbs, Math.abs(value)));
  if (maxAbs <= 0) {
    return 1;
  }
  return Math.max(1, Math.min(10, 0.82 / maxAbs));
}

function scalePositions(positions: number[], scale: number) {
  for (let i = 0; i < positions.length; i++) {
    positions[i] *= scale;
  }
}

function getRegionsByWalkStep(): RandomWalkRegion[] {
  return randomWalkRegions.slice().sort((a, b) => a.walkStep - b.walkStep);
}

function getModelOutputVector(point: Example2D): number[] {
  if (network == null) {
    return point.labelVec || [point.label];
  }
  try {
    return nn.forwardPropOutputs(network, constructInput(point));
  } catch (e) {
    return point.labelVec || [point.label];
  }
}

function addPosition(result: number[], values: number[], axes: number[],
    range: number) {
  result.push(normalizeProjectionValue(values[axes[0]] || 0, range),
      normalizeProjectionValue(values[axes[1]] || 0, range),
      normalizeProjectionValue(values[axes[2]] || 0, range));
}

function valuesToColors(values: number[], scale): number[] {
  let colors: number[] = [];
  values.forEach(value => {
    let color = d3.rgb(scale(value) as any);
    colors.push(color.r / 255, color.g / 255, color.b / 255);
  });
  return colors;
}

function solidColors(count: number, colorValue: string): number[] {
  let colors: number[] = [];
  let color = d3.rgb(colorValue as any);
  for (let i = 0; i < count; i++) {
    colors.push(color.r / 255, color.g / 255, color.b / 255);
  }
  return colors;
}

function makeMseColorScale(values: number[]) {
  let maxError = Math.max(maxFinite(values), 1e-6);
  return d3.scale.linear<string, number>()
      .domain([0, maxError * 0.5, maxError])
      .range(["#2a9d8f", "#f4d35e", "#d1495b"])
      .clamp(true);
}

function maxFinite(values: number[]): number {
  let result = 0;
  values.forEach(value => {
    if (isFinite(value)) {
      result = Math.max(result, value);
    }
  });
  return result;
}

function makeRandomWalkValueColorScale(values: number[]) {
  let maxAbs = 0;
  values.forEach(value => {
    if (isFinite(value)) {
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  });
  maxAbs = Math.max(maxAbs, 1 / Math.max(2, state.walkLengthK));
  return d3.scale.linear<string, number>()
      .domain([-maxAbs, 0, maxAbs])
      .range(["#f59322", "#e8eaeb", "#0877bd"])
      .clamp(true);
}

function buildOutputSpaceOodPlanePayload(range: number,
    modelVectors: number[][]): OodPlanePayload {
  if (!state.showOodPlane || modelVectors.length === 0) {
    return null;
  }
  let axes = getProjectionAxes();
  let fixedAxis = Math.max(0, Math.min(2, Math.floor(state.oodPlaneAxis)));
  let freeAxes: number[] = [];
  for (let axis = 0; axis < 3; axis++) {
    if (axis !== fixedAxis) {
      freeAxes.push(axis);
    }
  }
  let offset = Math.max(-1, Math.min(1, +state.oodPlaneOffset || 0));
  let positions: number[] = [];
  let errors: number[] = [];
  let indices: number[] = [];
  let resolution = OOD_PLANE_RESOLUTION;
  for (let row = 0; row <= resolution; row++) {
    let v = -1 + 2 * row / resolution;
    for (let col = 0; col <= resolution; col++) {
      let u = -1 + 2 * col / resolution;
      let projected = [0, 0, 0];
      projected[fixedAxis] = offset;
      projected[freeAxes[0]] = u;
      projected[freeAxes[1]] = v;
      positions.push(projected[0], projected[1], projected[2]);
      let targetOutput = zeroVector(getOutputDimension());
      targetOutput[axes[0]] = projected[0] * range;
      targetOutput[axes[1]] = projected[1] * range;
      targetOutput[axes[2]] = projected[2] * range;
      errors.push(getNearestPredictionMse(targetOutput, modelVectors));
    }
  }
  for (let row = 0; row < resolution; row++) {
    for (let col = 0; col < resolution; col++) {
      let a = row * (resolution + 1) + col;
      let b = a + 1;
      let c = a + resolution + 1;
      let d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  return {
    positions,
    colors: valuesToColors(errors, makeMseColorScale(errors)),
    indices,
    maxError: maxFinite(errors)
  };
}

function getNearestPredictionMse(targetOutput: number[],
    modelVectors: number[][]): number {
  let bestError = Infinity;
  modelVectors.forEach(output => {
    bestError = Math.min(bestError, getVectorMse(output, targetOutput));
  });
  return isFinite(bestError) ? bestError : 0;
}

function getVectorMse(output: number[], target: number[]): number {
  let length = Math.max(1, Math.min(output.length, target.length));
  let error = 0;
  for (let i = 0; i < length; i++) {
    let diff = (output[i] || 0) - (target[i] || 0);
    error += diff * diff;
  }
  return error / length;
}

function getGroundTruthValue(point: Example2D): number {
  if (point.regionTargetVec != null && point.regionTargetVec.length > 0) {
    return point.regionTargetVec[0];
  }
  if (point.labelVec != null && point.labelVec.length > 0) {
    return point.labelVec[0];
  }
  return point.label;
}

function exampleFromRegion(region: RandomWalkRegion): Example2D {
  return {
    x: region.center[0],
    y: region.center.length > 1 ? region.center[1] : 0,
    label: region.targetVec[0],
    inputs: region.center.slice(),
    labelVec: region.targetVec.slice(),
    regionIndex: region.index,
    regionCenter: region.center.slice(),
    regionTargetVec: region.targetVec.slice()
  };
}

function getProjectionAxes(): number[] {
  let maxIndex = getOutputDimension() - 1;
  return [
    Math.min(state.projectionX, maxIndex),
    Math.min(state.projectionY, maxIndex),
    Math.min(state.projectionZ, maxIndex)
  ];
}

function getProjectionAxisLabels(): string[] {
  return getProjectionAxes().map(axis => "X" + (axis + 1));
}

function createPointCloud(positions: number[], colors: number[], size: number,
    opacity: number): any {
  let geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(
      positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return new THREE.Points(geometry, new THREE.PointsMaterial({
    size,
    vertexColors: true,
    transparent: true,
      opacity
  }));
}

function createColoredSurface(positions: number[], colors: number[],
    indices: number[], opacity: number): any {
  let geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(
      positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    side: THREE.DoubleSide,
    vertexColors: true,
    transparent: true,
    opacity,
    depthWrite: false
  }));
}

function createSegmentedLine(positions: number[], colors: number[],
    radius: number, opacity: number): any {
  let group = new THREE.Group();
  let up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < positions.length - 3; i += 3) {
    let start = new THREE.Vector3(positions[i], positions[i + 1],
        positions[i + 2]);
    let end = new THREE.Vector3(positions[i + 3], positions[i + 4],
        positions[i + 5]);
    let direction = new THREE.Vector3().subVectors(end, start);
    let length = direction.length();
    if (length <= 1e-6) {
      continue;
    }
    let color = new THREE.Color(
        (colors[i] + colors[i + 3]) / 2,
        (colors[i + 1] + colors[i + 4]) / 2,
        (colors[i + 2] + colors[i + 5]) / 2);
    let geometry = new THREE.CylinderGeometry(radius, radius, length, 10);
    let material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity
    });
    let segment = new THREE.Mesh(geometry, material);
    segment.position.copy(start).add(end).multiplyScalar(0.5);
    segment.quaternion.setFromUnitVectors(up, direction.normalize());
    group.add(segment);
  }
  return group;
}

function clearProjectionDynamicObjects(view: ProjectionViewState) {
  if (view.root == null) {
    return;
  }
  view.dynamicObjects.forEach(object => view.root.remove(object));
  view.dynamicObjects = [];
  view.points = null;
}

function addProjectionObject(view: ProjectionViewState, object: any) {
  view.root.add(object);
  view.dynamicObjects.push(object);
}

function getProjectionColorValue(point: Example2D,
    useNetworkOutput: boolean): number {
  if (selectedNodeId != null) {
    return getNodeProjectionValue(point, selectedNodeId);
  }
  if (useNetworkOutput) {
    return getModelOutputValue(point, 0);
  }
  return getGroundTruthValue(point);
}

function getModelOutputValue(point: Example2D, outputIndex: number): number {
  if (network == null) {
    return point.label;
  }
  try {
    let output = nn.forwardPropOutputs(network, constructInput(point));
    return output[outputIndex] == null ? point.label : output[outputIndex];
  } catch (e) {
    return point.label;
  }
}

function ensureProjectionScene(name: string, container: HTMLDivElement) {
  let view = getProjectionView(name);
  let width = container.clientWidth || 720;
  let height = container.clientHeight || 420;
  if (view.renderer != null) {
    let canvas = view.renderer.domElement as HTMLCanvasElement;
    if (canvas.width !== width || canvas.height !== height) {
      view.camera.aspect = width / height;
      view.camera.updateProjectionMatrix();
      view.renderer.setSize(width, height);
    }
    return;
  }
  if (view.webglFailed) {
    return;
  }
  try {
    view.scene = new THREE.Scene();
    view.scene.background = new THREE.Color(0xfafafa);
    view.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 50);
    view.camera.position.set(0, 0, 3.4);
    view.renderer = new THREE.WebGLRenderer({antialias: true});
    view.renderer.setSize(width, height);
    container.innerHTML = "";
    container.appendChild(view.renderer.domElement);
    view.root = new THREE.Group();
    view.scene.add(view.root);
    view.root.add(new THREE.AxesHelper(name === "output" ? 1.05 : 1.15));
    attachProjectionMouseControls(name, view.renderer.domElement);
  } catch (e) {
    view.webglFailed = true;
    view.renderer = null;
  }
}

function attachProjectionMouseControls(name: string, canvas: HTMLCanvasElement) {
  let view = getProjectionView(name);
  if (view.controlsAttached) {
    return;
  }
  view.controlsAttached = true;
  canvas.className += " projection-orbit-canvas";
  canvas.setAttribute("title",
      "Drag to orbit. Ctrl/Alt-drag rolls. Wheel zooms all 3D views.");
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("mousedown", event => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    event.preventDefault();
  });
  window.addEventListener("mousemove", event => {
    if (!dragging) {
      return;
    }
    let dx = event.clientX - lastX;
    let dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    if (event.ctrlKey || event.altKey) {
      state.projectionRotateZ = normalizeAngle(state.projectionRotateZ +
          dx * 0.45);
      state.projectionRotateX = normalizeAngle(state.projectionRotateX +
          dy * 0.2);
    } else {
      state.projectionRotateY = normalizeAngle(state.projectionRotateY +
          dx * 0.45);
      state.projectionRotateX = normalizeAngle(state.projectionRotateX +
          dy * 0.45);
    }
    syncProjectionControlValues();
    renderAllProjectionViews();
    event.preventDefault();
  });
  window.addEventListener("mouseup", () => {
    if (dragging) {
      dragging = false;
      state.serialize();
    }
  });
  canvas.addEventListener("wheel", event => {
    let nextScale = state.projectionScale *
        Math.exp(-event.deltaY * 0.0012);
    state.projectionScale = Math.max(0.5, Math.min(2.5, nextScale));
    syncProjectionControlValues();
    renderAllProjectionViews();
    state.serialize();
    event.preventDefault();
  });
}

function normalizeAngle(value: number): number {
  while (value > 180) {
    value -= 360;
  }
  while (value < -180) {
    value += 360;
  }
  return value;
}

function syncProjectionControlValues() {
  ["projectionRotateX", "projectionRotateY", "projectionRotateZ",
      "projectionScale"].forEach(prop => {
    d3.select("#" + prop).property("value", state[prop]);
  });
}

function renderAllProjectionViews() {
  for (let name in projectionViews) {
    renderProjectionView(name);
  }
}

function getProjectionView(name: string): ProjectionViewState {
  if (projectionViews[name] == null) {
    projectionViews[name] = {
      scene: null,
      camera: null,
      renderer: null,
      root: null,
      points: null,
      dynamicObjects: [],
      webglFailed: false,
      controlsAttached: false
    };
  }
  return projectionViews[name];
}

function renderProjectionView(name: string) {
  let view = projectionViews[name];
  if (view == null || view.renderer == null || view.scene == null ||
      view.camera == null || view.root == null) {
    return;
  }
  view.root.rotation.x = state.projectionRotateX * Math.PI / 180;
  view.root.rotation.y = state.projectionRotateY * Math.PI / 180;
  view.root.rotation.z = state.projectionRotateZ * Math.PI / 180;
  view.root.scale.set(state.projectionScale, state.projectionScale,
      state.projectionScale);
  view.renderer.render(view.scene, view.camera);
}

function drawFallbackProjection(container: HTMLDivElement, positions: number[],
    colors: number[]) {
  let width = container.clientWidth || 720;
  let height = container.clientHeight || 420;
  let canvas = container.querySelector("canvas") as HTMLCanvasElement;
  if (canvas == null) {
    container.innerHTML = "";
    canvas = document.createElement("canvas");
    container.appendChild(canvas);
  }
  canvas.width = width;
  canvas.height = height;
  let context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fafafa";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#ddd";
  context.strokeRect(0, 0, width, height);
  let scale = Math.min(width, height) * 0.42 * state.projectionScale;
  let rx = state.projectionRotateX * Math.PI / 180;
  let ry = state.projectionRotateY * Math.PI / 180;
  let rz = state.projectionRotateZ * Math.PI / 180;
  for (let i = 0; i < positions.length; i += 3) {
    let rotated = rotatePoint(positions[i], positions[i + 1],
        positions[i + 2], rx, ry, rz);
    let colorIndex = i;
    context.fillStyle = "rgb(" +
        Math.round(colors[colorIndex] * 255) + "," +
        Math.round(colors[colorIndex + 1] * 255) + "," +
        Math.round(colors[colorIndex + 2] * 255) + ")";
    context.beginPath();
    context.arc(width / 2 + rotated[0] * scale,
        height / 2 - rotated[1] * scale, 2.3, 0, Math.PI * 2);
    context.fill();
  }
}

function getProjectionRange(): number {
  return Math.max(0.1, +state.projectionRange || 1);
}

function normalizeProjectionValue(value: number, range: number): number {
  return Math.max(-1, Math.min(1, value / range));
}

function rotatePoint(x: number, y: number, z: number, rx: number, ry: number,
    rz: number): number[] {
  let cosX = Math.cos(rx), sinX = Math.sin(rx);
  let y1 = y * cosX - z * sinX;
  let z1 = y * sinX + z * cosX;
  let cosY = Math.cos(ry), sinY = Math.sin(ry);
  let x2 = x * cosY + z1 * sinY;
  let z2 = -x * sinY + z1 * cosY;
  let cosZ = Math.cos(rz), sinZ = Math.sin(rz);
  return [x2 * cosZ - y1 * sinZ, x2 * sinZ + y1 * cosZ, z2];
}

function getNodeProjectionValue(point: Example2D, nodeId: string): number {
  if (network == null) {
    return point.label;
  }
  try {
    nn.forwardProp(network, constructInput(point));
  } catch (e) {
    return point.label;
  }
  let node = findNode(network, nodeId);
  return node == null ? point.label : node.output;
}

function findNode(network: nn.Node[][], nodeId: string): nn.Node {
  for (let layerIdx = 0; layerIdx < network.length; layerIdx++) {
    let layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      if (layer[i].id === nodeId) {
        return layer[i];
      }
    }
  }
  return null;
}

function updateDeadNeuronMonitor() {
  let container = d3.select("#dead-neuron-stats");
  if (!container.size() || network == null) {
    return;
  }
  let rows: string[] = [];
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let layer = network[layerIdx];
    let dead = 0;
    for (let i = 0; i < layer.length; i++) {
      let node = layer[i];
      if (isNeuronUnhealthy(node)) {
        dead++;
      }
    }
    rows.push("L" + layerIdx + ": " + dead + "/" + layer.length);
  }
  container.text(rows.join("  "));
}

function isNeuronUnhealthy(node: nn.Node): boolean {
  let eps = state.deadNeuronEps;
  if (node.activation === nn.Activations.RELU) {
    return Math.abs(node.output) <= eps;
  }
  if (node.activation === nn.Activations.SIGMOID) {
    return node.output <= eps || node.output >= 1 - eps;
  }
  if (node.activation === nn.Activations.TANH) {
    return node.output <= -1 + eps || node.output >= 1 - eps;
  }
  return Math.abs(node.output) <= eps;
}

function runAutoSweep(kind: string) {
  if (!usingRandomWalk()) {
    d3.select("#sweep-status").text("Switch to random-walk regression first.");
    return;
  }
  d3.select("#sweep-status").text("Running " + kind + " sweep...");
  let originalK = state.walkLengthK;
  let originalSeed = state.seed;
  let ks = [2, 3, 4, 5].filter(k => k <= getMaxRandomWalkK());
  let tbody = d3.select("#sweep-results tbody");
  tbody.selectAll("tr").remove();
  let results: any[] = [];
  ks.forEach(k => {
    state.walkLengthK = clampRandomWalkK(k);
    Math.seedrandom(originalSeed);
    let data = randomWalkRegressionData({
      dimension: state.dimensionN,
      walkLength: state.walkLengthK,
      sampleMultiplier: state.sampleMultiplierM,
      noiseEnabled: state.noiseEnabled,
      noiseMean: state.noiseMean,
      noiseVariance: state.noiseVariance
    });
    shuffle(data);
    let splitIndex = Math.floor(data.length * 0.8);
    let localTrain = data.slice(0, splitIndex);
    let localTest = data.slice(splitIndex);
    let values = kind === "depth" ? [2, 4, 6, 8] : [4, 8, 16, 32];
    values.forEach(value => {
      let depth = kind === "depth" ? value : 4;
      let width = kind === "width" ? value : 16;
      let shape = d3.range(depth).map(() => width);
      let localNetwork = buildSweepNetwork(shape);
      for (let epoch = 0; epoch < state.sweepEpochs; epoch++) {
        trainOneEpoch(localNetwork, localTrain);
      }
      results.push({
        kind,
        k,
        depth,
        width,
        trainLoss: getLoss(localNetwork, localTrain),
        testLoss: getLoss(localNetwork, localTest)
      });
    });
  });
  state.walkLengthK = originalK;
  state.seed = originalSeed;
  renderSweepResults(results);
  d3.select("#sweep-status").text("Sweep complete.");
}

function buildSweepNetwork(hiddenShape: number[]): nn.Node[][] {
  let inputIds = constructInputIds();
  let shape = [inputIds.length].concat(hiddenShape)
      .concat([getOutputDimension()]);
  return nn.buildNetwork(shape, state.activation, nn.Activations.LINEAR,
      state.regularization, inputIds, state.initZero);
}

function trainOneEpoch(localNetwork: nn.Node[][], data: Example2D[]) {
  data.forEach((point, i) => {
    nn.forwardProp(localNetwork, constructInput(point));
    nn.backProp(localNetwork, getTarget(point), nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(localNetwork, state.learningRate,
          state.regularizationRate);
    }
  });
}

function renderSweepResults(results: any[]) {
  let rows = d3.select("#sweep-results tbody").selectAll("tr").data(results);
  let entered = rows.enter().append("tr");
  ["kind", "k", "depth", "width", "trainLoss", "testLoss"].forEach(key => {
    entered.append("td").attr("class", key);
  });
  rows.select("td.kind").text(d => d.kind);
  rows.select("td.k").text(d => d.k);
  rows.select("td.depth").text(d => d.depth);
  rows.select("td.width").text(d => d.width);
  rows.select("td.trainLoss").text(d => d.trainLoss.toFixed(4));
  rows.select("td.testLoss").text(d => d.testLoss.toFixed(4));
  rows.exit().remove();
  drawSweepHeatmap(results);
}

function drawSweepHeatmap(results: any[]) {
  let canvas = document.querySelector("#sweep-heatmap") as HTMLCanvasElement;
  if (canvas == null) {
    return;
  }
  let context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#fafafa";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (results.length === 0) {
    return;
  }
  let ks = uniqueSorted(results.map(r => r.k));
  let axisName = results[0].kind === "depth" ? "depth" : "width";
  let capacities = uniqueSorted(results.map(r => r[axisName]));
  let losses = results.map(r => r.testLoss);
  let minLoss = Math.min.apply(null, losses);
  let maxLoss = Math.max.apply(null, losses);
  let left = 36;
  let top = 12;
  let right = 8;
  let bottom = 26;
  let plotWidth = canvas.width - left - right;
  let plotHeight = canvas.height - top - bottom;
  let cellWidth = plotWidth / ks.length;
  let cellHeight = plotHeight / capacities.length;

  context.font = "10px sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  results.forEach(result => {
    let xIndex = ks.indexOf(result.k);
    let yIndex = capacities.indexOf(result[axisName]);
    let normalized = maxLoss === minLoss ? 0 : (result.testLoss - minLoss) /
        (maxLoss - minLoss);
    context.fillStyle = d3.interpolateRgb("#e8eaeb", "#0877bd")(1 - normalized);
    context.fillRect(left + xIndex * cellWidth, top + yIndex * cellHeight,
        cellWidth, cellHeight);
    context.fillStyle = "#222";
    context.fillText(result.testLoss.toFixed(2),
        left + xIndex * cellWidth + cellWidth / 2,
        top + yIndex * cellHeight + cellHeight / 2);
  });

  context.strokeStyle = "#ddd";
  context.strokeRect(left, top, plotWidth, plotHeight);
  context.fillStyle = "#333";
  ks.forEach((k, i) => {
    context.fillText("K" + k, left + i * cellWidth + cellWidth / 2,
        top + plotHeight + 12);
  });
  context.textAlign = "right";
  capacities.forEach((capacity, i) => {
    context.fillText(String(capacity), left - 6,
        top + i * cellHeight + cellHeight / 2);
  });
  context.textAlign = "left";
  context.fillText(axisName, 4, 8);
}

function uniqueSorted(values: number[]): number[] {
  let seen: {[key: string]: boolean} = {};
  let result: number[] = [];
  values.forEach(value => {
    let key = String(value);
    if (!seen[key]) {
      seen[key] = true;
      result.push(value);
    }
  });
  return result.sort((a, b) => a - b);
}

function constructInputIds(): string[] {
  let rawSize = usingRandomWalk() ?
      getOutputDimension() : constructRawInput(0, 0).length;
  let frontendSize = getFrontendOutputSize(rawSize);
  inputLabels = {};
  if (state.useTransformer) {
    let ids = buildFeatureIds("t", frontendSize);
    ids.forEach((id, i) => inputLabels[id] = "T_" + (i + 1));
    return ids;
  }
  if (state.useCnnFrontend) {
    let ids = buildFeatureIds("c", frontendSize);
    ids.forEach((id, i) => inputLabels[id] = "C_" + (i + 1));
    return ids;
  }
  if (usingRandomWalk()) {
    let ids = buildFeatureIds("x", rawSize);
    ids.forEach((id, i) => inputLabels[id] = "X_" + (i + 1));
    return ids;
  }
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function buildFeatureIds(prefix: string, count: number): string[] {
  let ids: string[] = [];
  for (let i = 0; i < count; i++) {
    ids.push(prefix + (i + 1));
  }
  return ids;
}

function constructInput(xOrPoint: number | Example2D, y?: number): number[] {
  return applyArchitectureFrontend(constructRawInput(xOrPoint, y));
}

function constructRawInput(xOrPoint: number | Example2D, y?: number): number[] {
  if (typeof xOrPoint !== "number") {
    let point = xOrPoint as Example2D;
    if (point.inputs != null) {
      return point.inputs.slice();
    }
    return constructRawInput(point.x, point.y);
  }
  let x = xOrPoint as number;
  if (usingRandomWalk()) {
    let input: number[] = [];
    for (let i = 0; i < getOutputDimension(); i++) {
      input.push(i === 0 ? x : (i === 1 ? y : 0));
    }
    return input;
  }
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function applyArchitectureFrontend(input: number[]): number[] {
  if (state.useTransformer) {
    return transformerFrontend(input);
  }
  if (state.useCnnFrontend) {
    return cnnFrontend(input);
  }
  return input;
}

function cnnFrontend(input: number[]): number[] {
  let filters = 8;
  let kernelSize = input.length < 2 ? 1 : 2;
  let positions = Math.max(1, input.length - kernelSize + 1);
  let result: number[] = [];
  for (let filter = 0; filter < filters; filter++) {
    let sum = 0;
    for (let position = 0; position < positions; position++) {
      let z = frontendWeight(filter, 0, 0) * 0.1;
      for (let k = 0; k < kernelSize; k++) {
        z += input[position + k] * frontendWeight(filter, k + 1, 0);
      }
      sum += z;
    }
    result.push(sum / positions);
  }
  return result;
}

function transformerFrontend(input: number[]): number[] {
  let dModel = 16;
  let heads = 4;
  let headDim = dModel / heads;
  let tokens = input.map((value, tokenIndex) => {
    let embedding: number[] = [];
    for (let d = 0; d < dModel; d++) {
      embedding.push(value * frontendWeight(tokenIndex, d, 1) +
          0.1 * Math.sin((tokenIndex + 1) * (d + 1)));
    }
    return embedding;
  });
  if (tokens.length === 0) {
    let emptyToken: number[] = [];
    for (let d = 0; d < dModel; d++) {
      emptyToken.push(0);
    }
    tokens.push(emptyToken);
  }
  let encoded = tokens.map((token, tokenIndex) => {
    let context: number[] = [];
    for (let head = 0; head < heads; head++) {
      let scores = tokens.map((other, otherIndex) => {
        let score = 0;
        for (let d = 0; d < headDim; d++) {
          let modelIndex = head * headDim + d;
          let q = token[modelIndex] * frontendWeight(head, modelIndex, 2);
          let k = other[modelIndex] * frontendWeight(head, modelIndex, 3);
          score += q * k;
        }
        return score / Math.sqrt(headDim) +
            (tokenIndex === otherIndex ? 0.05 : 0);
      });
      let weights = softmax(scores);
      for (let d = 0; d < headDim; d++) {
        let modelIndex = head * headDim + d;
        let value = 0;
        for (let otherIndex = 0; otherIndex < tokens.length; otherIndex++) {
          value += weights[otherIndex] * tokens[otherIndex][modelIndex] *
              frontendWeight(head, modelIndex, 4);
        }
        context.push(value);
      }
    }
    let ffn: number[] = [];
    for (let d = 0; d < dModel; d++) {
      let hidden = Math.max(0, context[d] * frontendWeight(d, 0, 5) +
          frontendWeight(d, 1, 5) * 0.1);
      ffn.push(context[d] + hidden * frontendWeight(d, 2, 5));
    }
    return ffn;
  });
  let pooled = new Array(dModel);
  for (let d = 0; d < dModel; d++) {
    let sum = 0;
    for (let i = 0; i < encoded.length; i++) {
      sum += encoded[i][d];
    }
    pooled[d] = sum / encoded.length;
  }
  return pooled;
}

function softmax(values: number[]): number[] {
  let max = Math.max.apply(null, values);
  let exps = values.map(value => Math.exp(value - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(value => value / sum);
}

function frontendWeight(a: number, b: number, c: number): number {
  return Math.sin((a + 1) * 12.9898 + (b + 1) * 78.233 +
      (c + 1) * 37.719);
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point);
    nn.forwardProp(network, input);
    nn.backProp(network, getTarget(point), nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let inputIds = constructInputIds();
  let numInputs = inputIds.length;
  let shape = [numInputs].concat(state.networkShape)
      .concat([getOutputDimension()]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      state.regularization, inputIds, state.initZero);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    let randomWalkCanvas: any =
        document.querySelector("#random-walk-dataset");
    if (randomWalkCanvas != null) {
      renderThumbnail(randomWalkCanvas, function() {
        return randomWalkRegressionData({
          dimension: Math.max(2, state.dimensionN),
          walkLength: state.walkLengthK,
          sampleMultiplier: 10,
          noiseEnabled: false,
          noiseMean: 0,
          noiseVariance: 0
        });
      });
    }
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  if (d3.select("#seed").size()) {
    d3.select("#seed").property("value", state.seed);
  }
  let data: Example2D[];
  if (usingRandomWalk()) {
    data = randomWalkRegressionData({
      dimension: state.dimensionN,
      walkLength: state.walkLengthK,
      sampleMultiplier: state.sampleMultiplierM,
      noiseEnabled: state.noiseEnabled,
      noiseMean: state.noiseMean,
      noiseVariance: state.noiseVariance
    });
    let generatedData: any = data;
    randomWalkRegions = generatedData.randomWalkRegions || [];
    randomWalkPath = generatedData.randomWalkPath || [];
  } else {
    randomWalkRegions = [];
    randomWalkPath = [];
    let numSamples = (state.problem === Problem.REGRESSION) ?
        NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    let generator = state.problem === Problem.CLASSIFICATION ?
        state.dataset : state.regDataset;
    data = generator(numSamples, state.noise / 100);
  }
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
  updateProjectionControls();
  drawRandomWalkProjection();
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
