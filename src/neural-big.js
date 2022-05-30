const { Encoder } = require('@agarimo/encoder');
const { LRUCache } = require('@agarimo/lru-cache');
const { runMulti } = require('./neural-multi');

const defaultSettings = require('./default-settings.json');

const defaultLogFn = (status, time) =>
  console.log(`Epoch ${status.iterations} loss ${status.error} time ${time}ms`);

class Neural {
  constructor(settings = {}) {
    this.settings = { ...settings };
    this.settings = { ...settings };
    Object.keys(defaultSettings).forEach((key) => {
      if (this.settings[key] === undefined) {
        this.settings[key] = defaultSettings[key];
      }
    });
    if (this.settings.log === true) {
      this.logFn = defaultLogFn;
    } else if (typeof this.settings.log === 'function') {
      this.logFn = this.settings.log;
    }
  }

  prepareCorpus(corpus) {
    if (this.settings.encoder) {
      this.encoder = this.settings.encoder;
    } else {
      this.encoder = new Encoder({
        processor: this.settings.processor,
        unknokwnIndex: this.settings.unknokwnIndex,
        useCache: this.settings.useCache,
        cacheSize: this.settings.cacheSize,
      });
    }
    this.encoded = this.encoder.encodeCorpus(corpus);
  }

  initialize() {
    this.useCache =
      this.settings.useCache === undefined ? true : this.settings.useCache;
    this.cacheSize = this.settings.cacheSize || 10000;
    if (this.useCache) {
      this.cache = new LRUCache(this.cacheSize);
    }
    const labels = this.encoder.intents;
    this.perceptrons = [];
    this.perceptronByName = {};
    this.numPerceptrons = labels.length;
    for (let i = 0; i < this.numPerceptrons; i += 1) {
      const name = labels[i];
      const perceptron = {
        name,
        id: i,
        weights: new Float32Array(this.encoder.features.length),
        changes: new Float32Array(this.encoder.features.length),
        bias: 0,
      };
      this.perceptrons.push(perceptron);
      this.perceptronByName[name] = perceptron;
    }
  }

  runInputPerceptronTrain(perceptron, input) {
    const { weights, bias } = perceptron;
    let sum = bias;
    for (let i = 0; i < input.keys.length; i += 1) {
      const key = input.keys[i];
      sum += input.data[key] * weights[key];
    }
    return sum <= 0 ? 0 : this.settings.alpha * sum;
  }

  trainPerceptron(perceptron, data) {
    const { alpha, momentum } = this.settings;
    const { weights, changes } = perceptron;
    const dlr = this.decayLearningRate;
    let error = 0;
    for (let i = 0; i < data.length; i += 1) {
      const { input, output } = data[i];
      const actualOutput = this.runInputPerceptronTrain(perceptron, input);
      const expectedOutput = output.data[perceptron.id] || 0;
      const currentError = expectedOutput - actualOutput;
      if (currentError) {
        error += currentError ** 2;
        const delta = (actualOutput > 0 ? 1 : alpha) * currentError * dlr;
        for (let j = 0; j < input.keys.length; j += 1) {
          const key = input.keys[j];
          const change = delta * input.data[key] + momentum * changes[key];
          changes[key] = change;
          weights[key] += change;
        }
        // eslint-disable-next-line no-param-reassign
        perceptron.bias += delta;
      }
    }
    return error;
  }

  weightsToDict() {
    this.weightsDict = {};
    const numFeatures = this.encoder.features.length;
    for (let i = 0; i < numFeatures; i += 1) {
      for (let j = 0; j < this.numPerceptrons; j += 1) {
        if (this.perceptrons[j].weights[i] !== 0) {
          if (!this.weightsDict[i]) {
            this.weightsDict[i] = {};
          }
          this.weightsDict[i][j] = this.perceptrons[j].weights[i];
        }
      }
    }
    Object.keys(this.weightsDict).forEach((key) => {
      this.weightsDict[key] = {
        data: this.weightsDict[key],
        keys: Object.keys(this.weightsDict[key]),
      };
    });
  }

  removeWeightsAndChanges() {
    for (let i = 0; i < this.numPerceptrons; i += 1) {
      const perceptron = this.perceptrons[i];
      if (!this.settings.multi) {
        delete perceptron.weights;
      }
      delete perceptron.changes;
    }
  }

  resultsToClassifications(srcOutputs, validIntents) {
    const outputs = srcOutputs;
    const result = [];
    let total = 0;
    if (validIntents) {
      for (let i = 0; i < outputs.length; i += 1) {
        if (outputs[i].score > 0 && validIntents.includes(outputs[i].intent)) {
          outputs[i].score += this.perceptrons[i].bias;
          if (outputs[i].score > 0) {
            outputs[i].score = (outputs[i].score * this.settings.alpha) ** 2;
            total += outputs[i].score;
            result.push(outputs[i]);
          }
        }
      }
    } else {
      for (let i = 0; i < outputs.length; i += 1) {
        if (outputs[i].score > 0) {
          outputs[i].score += this.perceptrons[i].bias;
          if (outputs[i].score > 0) {
            outputs[i].score = (outputs[i].score * this.settings.alpha) ** 2;
            total += outputs[i].score;
            result.push(outputs[i]);
          }
        }
      }
    }
    if (result.length === 0) {
      return [{ intent: 'None', score: 1 }];
    }
    for (let i = 0; i < result.length; i += 1) {
      result[i].score /= total;
    }
    return result.sort((a, b) => b.score - a.score);
  }

  runInput(input, validIntents) {
    const outputs = this.perceptrons.map(({ name, bias }) => ({
      intent: name,
      score: bias,
    }));
    for (let i = 0, li = input.keys.length; i < li; i += 1) {
      const weights = this.weightsDict[input.keys[i]];
      if (weights) {
        for (let j = 0, lj = weights.keys.length; j < lj; j += 1) {
          const key = weights.keys[j];
          outputs[key].score += weights.data[key];
        }
      }
    }
    const result = [];
    let total = 0;
    if (validIntents) {
      for (let i = 0; i < outputs.length; i += 1) {
        if (outputs[i].score > 0 && validIntents.includes(outputs[i].intent)) {
          outputs[i].score = (outputs[i].score * this.settings.alpha) ** 2;
          total += outputs[i].score;
          result.push(outputs[i]);
        }
      }
    } else {
      for (let i = 0; i < outputs.length; i += 1) {
        if (outputs[i].score > 0) {
          outputs[i].score = (outputs[i].score * this.settings.alpha) ** 2;
          total += outputs[i].score;
          result.push(outputs[i]);
        }
      }
    }
    if (result.length === 0) {
      return [{ intent: 'None', score: 1 }];
    }
    for (let i = 0; i < result.length; i += 1) {
      result[i].score /= total;
    }
    return result.sort((a, b) => b.score - a.score);
    // return this.resultsToClassifications(outputs, validIntents);
  }

  run(text, validIntents) {
    let result;
    if (!validIntents && this.useCache) {
      result = this.cache.get(text);
      if (!result) {
        result = this.runInput(this.encoder.processText(text));
        this.cache.put(text, result);
      }
    }
    if (!result) {
      result = this.runInput(this.encoder.processText(text), validIntents);
    }
    if (!this.settings.multi) {
      return result;
    }
    return {
      monoIntent: result,
      multiIntent: runMulti(this, text, result[0].score, validIntents),
    };
  }

  train(corpus) {
    if (corpus) {
      const srcData = Array.isArray(corpus) ? corpus : corpus.data;
      if (!srcData || !srcData.length) {
        throw new Error('Invalid corpus received');
      }
      this.prepareCorpus(srcData);
      this.initialize();
    }
    if (!this.encoded || !this.encoded.train || !this.encoded.train.length) {
      throw new Error('Invalid corpus received');
    }
    const data = this.encoded.train;
    if (!this.status) {
      this.status = { error: Infinity, deltaError: Infinity, iterations: 0 };
    }
    const { errorThresh: minError, deltaErrorThresh: minDelta } = this.settings;
    while (
      this.status.iterations < this.settings.iterations &&
      this.status.error > minError &&
      this.status.deltaError > minDelta
    ) {
      const hrstart = new Date();
      this.status.iterations += 1;
      this.decayLearningRate =
        this.settings.learningRate / (1 + 0.001 * this.status.iterations);
      const lastError = this.status.error;
      this.status.error = 0;
      for (let i = 0; i < this.numPerceptrons; i += 1) {
        this.perceptrons[i].lastError = this.trainPerceptron(
          this.perceptrons[i],
          data
        );
        this.perceptrons[i].lastErrorOverData =
          this.perceptrons[i].lastError / data.length;
        this.status.error += this.perceptrons[i].lastError;
      }
      this.status.error /= this.numPerceptrons * data.length;
      this.status.deltaError = Math.abs(this.status.error - lastError);
      if (this.logFn) {
        const hrend = new Date();
        this.logFn(this.status, hrend.getTime() - hrstart.getTime());
      }
    }
    this.weightsToDict();
    if (!this.settings.keepWeightsAndChanges) {
      this.removeWeightsAndChanges();
    }
    return this.status;
  }

  measureCorpus(corpus) {
    let total = 0;
    let good = 0;
    for (let i = 0; i < corpus.data.length; i += 1) {
      const item = corpus.data[i];
      for (let j = 0; j < item.tests.length; j += 1) {
        const test = item.tests[j];
        const output = this.run(test);
        total += 1;
        const intent = Array.isArray(output) ? output[0].intent : output.intent;
        if (intent === item.intent) {
          good += 1;
        }
      }
    }
    return { good, total };
  }

  measure(corpus) {
    if (corpus) {
      return this.measureCorpus(corpus);
    }
    if (!this.encoded.validation || !(this.encoded.validation.length > 0)) {
      throw new Error('No corpus provided to measure');
    }
    let total = 0;
    let good = 0;
    for (let i = 0; i < this.encoded.validation.length; i += 1) {
      total += 1;
      const { input, output } = this.encoded.validation[i];
      const actual = this.runInput(input);
      const expectedIntent = this.encoder.getIntent(output.keys[0]);
      const actualIntent = actual[0].intent;
      if (expectedIntent === actualIntent) {
        good += 1;
      }
    }
    return { good, total };
  }

  toJSON(options = {}) {
    const result = {
      settings: { ...this.settings },
    };
    const weights = [];
    const keys = Object.keys(this.weightsDict);
    for (let i = 0; i < keys.length; i += 1) {
      weights.push(this.weightsDict[keys[i]].data);
    }
    result.weightsDict = weights;
    delete result.settings.processor;
    if (this.perceptrons) {
      result.perceptrons = [];
      for (let i = 0; i < this.perceptrons.length; i += 1) {
        const perceptron = this.perceptrons[i];
        const current = {
          name: perceptron.name,
          id: perceptron.id,
          weights: perceptron.weights ? [...perceptron.weights] : undefined,
          bias: perceptron.bias,
        };
        if (options.saveChanges) {
          current.changes = [...perceptron.changes];
        }
        result.perceptrons.push(current);
      }
      if (options.saveEncoder !== false) {
        result.encoder = this.encoder.toJSON();
      }
    }
    return result;
  }

  fromJSON(json) {
    this.settings = { ...this.settings, ...json.settings };
    this.weightsDict = {};
    for (let i = 0; i < json.weightsDict.length; i += 1) {
      this.weightsDict[i] = {
        data: json.weightsDict[i],
        keys: Object.keys(json.weightsDict[i]),
      };
    }
    if (json.encoder) {
      this.encoder = new Encoder({
        processor: this.settings.processor,
        unknokwnIndex: this.settings.unknokwnIndex,
        useCache: this.settings.useCache,
        cacheSize: this.settings.cacheSize,
      });
      this.encoder.fromJSON(json.encoder);
    }
    if (json.perceptrons) {
      this.initialize();
      for (let i = 0; i < json.perceptrons.length; i += 1) {
        const perceptron = json.perceptrons[i];
        const current = this.perceptronByName[perceptron.name];
        current.bias = perceptron.bias;
        if (perceptron.weights) {
          for (let j = 0; j < perceptron.weights.length; j += 1) {
            current.weights[j] = perceptron.weights[j];
          }
        }
        if (perceptron.changes) {
          for (let j = 0; j < perceptron.changes.length; j += 1) {
            current.changes[j] = perceptron.changes[j];
          }
        }
      }
    }
  }
}

module.exports = Neural;
