const Neural = require('./neural');
const NeuralBig = require('./neural-big');

function train(corpus, settings) {
  const net = new Neural(settings);
  net.train(corpus);
  return net;
}

function run(net, text, validIntents) {
  return net.run(text, validIntents);
}

function measure(net, corpus) {
  return net.measure(corpus);
}

module.exports = {
  Neural,
  NeuralBig,
  train,
  run,
  measure,
};
