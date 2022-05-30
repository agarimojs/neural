const { NeuralBig } = require('../src');
const corpus = require('./corpus-en.json');

describe('Neural Big', () => {
  describe('constructor', () => {
    it('Should create an instance with default settings', () => {
      const neural = new NeuralBig();
      expect(neural).toBeDefined();
      expect(neural.settings).toEqual({
        iterations: 20000,
        errorThresh: 0.00005,
        deltaErrorThresh: 0.000001,
        learningRate: 0.6,
        momentum: 0.5,
        alpha: 0.07,
        log: false,
        multi: false,
      });
    });
    it('Should allow to define partial settings', () => {
      const neural = new NeuralBig({ alpha: 0.08, log: true });
      expect(neural.settings).toEqual({
        iterations: 20000,
        errorThresh: 0.00005,
        deltaErrorThresh: 0.000001,
        learningRate: 0.6,
        momentum: 0.5,
        alpha: 0.08,
        log: true,
        multi: false,
      });
    });
  });

  describe('train', () => {
    test('It should train during several epochs', () => {
      const net = new NeuralBig();
      const status = net.train(corpus);
      expect(status.iterations).toEqual(34);
    });
  });

  describe('measure', () => {
    test('It should measure the validation corpus', () => {
      const net = new NeuralBig();
      net.train(corpus);
      const { good, total } = net.measure();
      expect(total).toEqual(256);
      expect(good).toEqual(202);
    });
  });

  describe('measure corpus', () => {
    test('It should measure an external corpus', () => {
      const net = new NeuralBig();
      net.train(corpus);
      const { good, total } = net.measureCorpus(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(202);
    });
  });

  describe('run', () => {
    test('It can use mono-intent detection', () => {
      const net = new NeuralBig();
      net.train(corpus);
      const input = 'who are you, when is your birthday and who is your boss?';
      const actual = net.run(input);
      const expected = [
        {
          intent: 'smalltalk.boss',
          score: 0.7464177512679069,
        },
        {
          intent: 'smalltalk.birthday',
          score: 0.2535210691874059,
        },
        {
          intent: 'smalltalk.right',
          score: 0.00006117954468731187,
        },
      ];
      expect(actual).toEqual(expected);
    });

    test('It can use multi-intent detection', () => {
      const net = new NeuralBig({ multi: true });
      net.train(corpus);
      const input = 'who are you, when is your birthday and who is your boss?';
      const actual = net.run(input);
      const expected = {
        monoIntent: [
          {
            intent: 'smalltalk.boss',
            score: 0.7464177512679069,
          },
          {
            intent: 'smalltalk.birthday',
            score: 0.2535210691874059,
          },
          {
            intent: 'smalltalk.right',
            score: 0.00006117954468731187,
          },
        ],
        multiIntent: [
          {
            tokens: ['who', 'are', 'you'],
            embeddings: [36, 64, 28],
            classifications: [
              {
                intent: 'smalltalk.acquaintance',
                score: 0.9994305864564091,
              },
              {
                intent: 'smalltalk.annoying',
                score: 0.00029939905400927795,
              },
              {
                intent: 'smalltalk.bad',
                score: 0.0002644902763842131,
              },
              {
                intent: 'smalltalk.hungry',
                score: 0.000005524213197446476,
              },
            ],
          },
          {
            tokens: ['when', 'is', 'your', 'birthday', 'and'],
            embeddings: [113, 8, 2, 114, 269],
            classifications: [
              {
                intent: 'smalltalk.birthday',
                score: 0.9874897374972665,
              },
              {
                intent: 'trivia.gc',
                score: 0.01251026250273345,
              },
            ],
          },
          {
            tokens: ['who', 'is', 'your', 'boss'],
            embeddings: [36, 8, 2, 141],
            classifications: [
              {
                intent: 'smalltalk.boss',
                score: 0.9999821681183945,
              },
              {
                intent: 'support.developers',
                score: 0.000017831881605606757,
              },
            ],
          },
        ],
      };
      expect(actual).toEqual(expected);
    });
  });

  describe('toJSON/fromJSON', () => {
    test('It should save to JSON and load from JSON', () => {
      const net = new NeuralBig();
      net.train(corpus);
      const json = net.toJSON();
      const net2 = new NeuralBig();
      net2.fromJSON(json);
      const { good, total } = net2.measure(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(202);
    });
    test('It can be saved/loaded including perceptron changes', () => {
      const net = new NeuralBig({ keepWeightsAndChanges: true });
      net.train(corpus);
      const json = net.toJSON({ saveChanges: true });
      const net2 = new NeuralBig();
      expect(json.perceptrons[0].changes).toBeDefined();
      net2.fromJSON(json);
      const { good, total } = net2.measure(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(202);
    });
    test('It can be chosen to do not save the encoder', () => {
      const net = new NeuralBig();
      net.train(corpus);
      const json = net.toJSON({ saveEncoder: false });
      const net2 = new NeuralBig();
      net2.prepareCorpus(corpus.data);
      net2.fromJSON(json);
      const { good, total } = net2.measure(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(202);
    });
  });
});
