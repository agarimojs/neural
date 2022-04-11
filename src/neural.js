const { Encoder } = require('@agarimo/encoder');
const { LRUCache } = require('@agarimo/lru-cache');
const { runMulti } = require('./neural-multi');
const defaultSettings = require('./default-settings.json');

const defaultLogFn = (status, time) =>
  console.log(`Epoch ${status.iterations} loss ${status.error} time ${time}ms`);

class Neural {
  constructor(settings = {}) {
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
    this.cacheSize = this.settings.cacheSize || 1000;
    if (this.useCache) {
      this.cache = new LRUCache(1000);
    }
    const labels = this.encoder.intents;
    const numFeatures = this.encoder.features.length;
    this.perceptrons = [];
    this.perceptronByName = {};
    this.numPerceptrons = labels.length;
    for (let i = 0; i < this.numPerceptrons; i += 1) {
      const name = labels[i];
      const perceptron = {
        name,
        id: i,
        weights: new Float32Array(numFeatures),
        changes: new Float32Array(numFeatures),
        bias: 0,
      };
      this.perceptrons.push(perceptron);
      this.perceptronByName[name] = perceptron;
    }
  }

  runInputPerceptronTrain(perceptron, input) {
    const sum = input.keys.reduce(
      (prev, key) => prev + input.data[key] * perceptron.weights[key],
      perceptron.bias
    );
    return sum <= 0 ? 0 : this.settings.alpha * sum;
  }

  runInputPerceptron(perceptron, input) {
    const sum = input.keys.reduce(
      (prev, key) => prev + input.data[key] * perceptron.weights[key],
      perceptron.bias
    );
    return sum <= perceptron.bias ? 0 : this.settings.alpha * sum;
  }

  runInput(input, validIntents) {
    const outputs = [];
    let total = 0;
    const perceptrons = validIntents
      ? validIntents.map((intent) => this.perceptronByName[intent])
      : this.perceptrons;
    for (let i = 0; i < perceptrons.length; i += 1) {
      const perceptron = perceptrons[i];
      const score = this.runInputPerceptron(perceptron, input);
      if (score > 0) {
        const item = { intent: perceptron.name, score: score ** 2 };
        outputs.push(item);
        total += item.score;
      }
    }
    if (total > 0) {
      for (let i = 0; i < outputs.length; i += 1) {
        outputs[i].score /= total;
      }
      return outputs.sort((a, b) => b.score - a.score);
    }
    return [{ intent: 'None', score: 1 }];
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
        this.status.error += this.trainPerceptron(this.perceptrons[i], data);
      }
      this.status.error /= this.numPerceptrons * data.length;
      this.status.deltaError = Math.abs(this.status.error - lastError);
      if (this.logFn) {
        const hrend = new Date();
        this.logFn(this.status, hrend.getTime() - hrstart.getTime());
      }
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
    delete result.settings.processor;
    if (this.perceptrons) {
      result.perceptrons = [];
      for (let i = 0; i < this.perceptrons.length; i += 1) {
        const perceptron = this.perceptrons[i];
        const current = {
          name: perceptron.name,
          id: perceptron.id,
          weights: [...perceptron.weights],
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
        for (let j = 0; j < perceptron.weights.length; j += 1) {
          current.weights[j] = perceptron.weights[j];
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
