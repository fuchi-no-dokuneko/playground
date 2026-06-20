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

import * as d3 from 'd3';

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number,
  y: number,
  label: number,
  inputs?: number[],
  labelVec?: number[],
  regionIndex?: number,
  regionCenter?: number[],
  regionTargetVec?: number[]
};

export interface RandomWalkRegion {
  index: number;
  walkStep: number;
  coords: number[];
  center: number[];
  targetVec: number[];
}

type Point = {
  x: number,
  y: number
};

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export interface RandomWalkOptions {
  dimension: number;
  walkLength: number;
  sampleMultiplier: number;
  noiseEnabled: boolean;
  noiseMean: number;
  noiseVariance: number;
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

export function randomWalkRegressionData(
    options: RandomWalkOptions): Example2D[] {
  let dimension = Math.max(1, Math.floor(options.dimension));
  let walkLength = Math.max(1, Math.floor(options.walkLength));
  let sampleMultiplier = Math.max(1, Math.floor(options.sampleMultiplier));
  let numSamples = walkLength * sampleMultiplier;
  let walk = generateRandomWalk(dimension, walkLength);
  let regionOrder = d3.range(walkLength);
  shuffle(regionOrder);
  let regions = buildRandomWalkRegions(dimension, walkLength, walk,
      regionOrder);
  let points: Example2D[] = [];

  for (let i = 0; i < numSamples; i++) {
    let regionIndex = Math.floor(Math.random() * walkLength);
    let inputs = sampleFromRegion(regionIndex, walkLength, dimension);
    let region = regions[regionIndex];
    let baseTarget = region.targetVec;
    let labelVec = baseTarget.map(value => {
      if (!options.noiseEnabled) {
        return value;
      }
      return value + normalRandom(options.noiseMean, options.noiseVariance);
    });
    points.push({
      x: inputs[0],
      y: dimension > 1 ? inputs[1] : 0,
      label: labelVec[0],
      inputs,
      labelVec,
      regionIndex,
      regionCenter: region.center.slice(),
      regionTargetVec: baseTarget.slice()
    });
  }
  (points as any).randomWalkRegions = regions;
  (points as any).randomWalkPath = walk.map(step => step.slice());
  return points;
}

function buildRandomWalkRegions(dimension: number, walkLength: number,
    walk: number[][], regionOrder: number[]): RandomWalkRegion[] {
  let regions: RandomWalkRegion[] = [];
  for (let regionIndex = 0; regionIndex < walkLength; regionIndex++) {
    let coords = regionCoordsFromIndex(regionIndex, walkLength, dimension);
    regions.push({
      index: regionIndex,
      walkStep: regionOrder[regionIndex],
      coords,
      center: regionCenterFromCoords(coords, walkLength, dimension),
      targetVec: walk[regionOrder[regionIndex]].slice()
    });
  }
  return regions;
}

function generateRandomWalk(dimension: number, walkLength: number): number[][] {
  let position = new Array(dimension);
  for (let i = 0; i < dimension; i++) {
    position[i] = 0;
  }
  let result: number[][] = [];
  for (let step = 0; step < walkLength; step++) {
    let dim = Math.floor(Math.random() * dimension);
    let direction = Math.random() < 0.5 ? -1 : 1;
    position[dim] += direction / walkLength;
    result.push(position.slice());
  }
  return result;
}

function sampleFromRegion(
    regionIndex: number, numRegions: number, dimension: number): number[] {
  let side = randomWalkRegionSide(numRegions, dimension);
  let cellSize = 2 / side;
  let coords = regionCoordsFromIndex(regionIndex, numRegions, dimension);
  return coords.map(coord => {
    let min = -1 + coord * cellSize;
    let max = Math.min(1, min + cellSize);
    return randUniform(min, max);
  });
}

function regionCoordsFromIndex(
    regionIndex: number, numRegions: number, dimension: number): number[] {
  let side = randomWalkRegionSide(numRegions, dimension);
  let coords = new Array(dimension);
  let remainder = regionIndex;
  for (let dim = 0; dim < dimension; dim++) {
    coords[dim] = remainder % side;
    remainder = Math.floor(remainder / side);
  }
  return coords;
}

function regionCenterFromCoords(
    coords: number[], numRegions: number, dimension: number): number[] {
  let side = randomWalkRegionSide(numRegions, dimension);
  let cellSize = 2 / side;
  let center = new Array(dimension);
  for (let dim = 0; dim < dimension; dim++) {
    center[dim] = -1 + coords[dim] * cellSize + cellSize / 2;
  }
  return center;
}

function randomWalkRegionSide(numRegions: number, dimension: number): number {
  return Math.ceil(Math.pow(numRegions, 1 / dimension));
}

export function classifyTwoGaussData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1); // Gaussian with positive examples.
  genGauss(-2, -2, -1); // Gaussian with negative examples.
  return points;
}

export function regressPlane(numSamples: number, noise: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scale.linear()
    .domain([-10, 10])
    .range([-1, 1]);
  let getLabel = (x, y) => labelScale(x + y);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x, y, label});
  }
  return points;
}

export function regressGaussian(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];

  let labelScale = d3.scale.linear()
    .domain([0, 2])
    .range([1, 0])
    .clamp(true);

  let gaussians = [
    [-4, 2.5, 1],
    [0, 2.5, -1],
    [4, 2.5, 1],
    [-4, -2.5, -1],
    [0, -2.5, 1],
    [4, -2.5, -1]
  ];

  function getLabel(x, y) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      let newLabel = sign * labelScale(dist({x, y}, {x: cx, y: cy}));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }
  let radius = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x, y, label});
  };
  return points;
}

export function classifySpiralData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number):
    Example2D[] {
  function getXORLabel(p: Point) { return p.x * p.y >= 0 ? 1 : -1; }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let padding = 0.3;
    x += x > 0 ? padding : -padding;  // Padding.
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-5, 5) * noise;
    let noiseY = randUniform(-5, 5) * noise;
    let label = getXORLabel({x: x + noiseX, y: y + noiseY});
    points.push({x, y, label});
  }
  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the eucledian distance between two points in space. */
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
