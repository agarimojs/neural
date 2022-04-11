function normalizeOutput(arr) {
  let total = 0;
  for (let i = 0; i < arr.length; i += 1) {
    const score = arr[i].score ** 2;
    total += score;
    // eslint-disable-next-line no-param-reassign
    arr[i].score = score;
  }
  if (total > 0) {
    for (let i = 0; i < arr.length; i += 1) {
      // eslint-disable-next-line no-param-reassign
      arr[i].score /= total;
    }
  }
}

function normalizeSlices(slices) {
  for (let i = 0; i < slices.length; i += 1) {
    normalizeOutput(slices[i].classifications);
  }
  return slices;
}

function calculateScore(input) {
  return input.length === 1
    ? input[0].score
    : input[0].score ** 2 - input[1].score ** 2;
}

function runInputPerceptronMulti(net, perceptron, input, keys) {
  let sum = perceptron.bias;
  const visited = {};
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    if (!visited[key]) {
      visited[key] = 1;
      sum += input.data[key] * perceptron.weights[key];
    }
  }
  return sum <= perceptron.bias ? 0 : net.settings.alpha * sum;
}

function runInputMulti(net, input, keys, validIntents) {
  const outputs = [];
  let total = 0;
  const perceptrons = validIntents
    ? validIntents.map((intent) => net.perceptronByName[intent])
    : net.perceptrons;
  for (let i = 0; i < perceptrons.length; i += 1) {
    const perceptron = perceptrons[i];
    const score = runInputPerceptronMulti(net, perceptron, input, keys);
    if (score > 0) {
      outputs.push({ intent: perceptron.name, score });
      total += score;
    }
  }
  if (total > 0) {
    return outputs.sort((a, b) => b.score - a.score);
  }
  return [{ intent: 'None', score: 1 }];
}

function runCached(net, input, keys, validIntents, cache = {}) {
  const strKey = keys.join('-');
  let result = cache[strKey];
  if (!result) {
    result = runInputMulti(net, input, keys, validIntents);
    // eslint-disable-next-line no-param-reassign
    cache[strKey] = result;
  }
  return result;
}

function getBestBinarySlices(net, data, keys, validIntents, cache = {}) {
  const input = { data, keys };
  const runOne = runCached(net, input, keys, validIntents, cache);
  let bestScore = calculateScore(runOne);
  let best = [keys];
  let runs = [runOne];
  for (let i = 1; i < keys.length; i += 1) {
    const left = keys.slice(0, i);
    const right = keys.slice(i);
    const runLeft = runCached(net, input, left, validIntents, cache);
    const runRight = runCached(net, input, right, validIntents, cache);
    const scoreLeft = calculateScore(runLeft);
    const scoreRight = calculateScore(runRight);
    const score = (scoreLeft + scoreRight) / 2;
    if (
      score > bestScore &&
      runLeft[0].score > 0.5 &&
      runRight[0].score > 0.5 &&
      runLeft[0].intent !== 'None' &&
      runRight[0].intent !== 'None'
    ) {
      bestScore = score;
      best = [left, right];
      runs = [runLeft, runRight];
    }
  }
  return best.length === 1
    ? [{ tokens: best[0], run: runs[0] }]
    : [
        { tokens: best[0], run: runs[0] },
        { tokens: best[1], run: runs[1] },
      ];
}

function getBestSlices(net, data, keys, validIntents, cache = {}) {
  const slices = getBestBinarySlices(net, data, keys, validIntents, cache);
  if (slices.length === 1) {
    return slices;
  }
  const left = getBestSlices(net, data, slices[0].tokens, validIntents, cache);
  const right = getBestSlices(net, data, slices[1].tokens, validIntents, cache);
  return [...left, ...right];
}

function runMulti(net, text, minScore, validIntents) {
  const tokens = net.encoder.processTextFull(text);
  const slices = getBestSlices(net, tokens.data, tokens.keys, validIntents);
  let total = 0;
  for (let i = 0; i < slices.length; i += 1) {
    const stems = [];
    const slice = slices[i];
    for (let j = 0; j < slice.tokens.length; j += 1) {
      stems.push(net.encoder.features[slice.tokens[j]]);
    }
    slice.embeddings = slice.tokens;
    slice.tokens = stems;
    slice.classifications = slice.run;
    total += slice.classifications[0].score;
    delete slice.run;
  }
  return total / slices.length < minScore ? [slices] : normalizeSlices(slices);
}

module.exports = {
  normalizeOutput,
  normalizeSlices,
  calculateScore,
  getBestBinarySlices,
  getBestSlices,
  runMulti,
};
