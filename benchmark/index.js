// eslint-disable-next-line import/no-extraneous-dependencies
const { Bench } = require('@agarimo/bench');
const corpus = require('../test/corpus-en.json');
const Neural = require('../src/neural');

function initWithCache() {
  const net = new Neural();
  net.train(corpus);
  return net;
}

function initWithoutCache() {
  const net = new Neural({ useCache: false });
  net.train(corpus);
  return net;
}

function measure(net) {
  return net.measure(corpus);
}

(async () => {
  const bench = new Bench({ transactionsPerRun: 256 });
  bench.add('With Cache', measure, initWithCache);
  bench.add('Without Cache', measure, initWithoutCache);
  const results = await bench.run();
  console.log(results);
})();
