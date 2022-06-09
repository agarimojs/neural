const { Neural, train, run, measure } = require('../src');
const corpus = require('./corpus-en.json');

describe('Neural', () => {
  describe('constructor', () => {
    it('Should create an instance with default settings', () => {
      const neural = new Neural();
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
      const neural = new Neural({ alpha: 0.08, log: true });
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
      const net = new Neural();
      const status = net.train(corpus);
      expect(status.iterations).toEqual(29);
    });
  });

  describe('measure', () => {
    test('It should measure the validation corpus', () => {
      const net = new Neural();
      net.train(corpus);
      const { good, total } = net.measure();
      expect(total).toEqual(256);
      expect(good).toEqual(203);
    });
  });

  describe('train and measure', () => {
    test('It should measure the validation corpus', () => {
      const net = train(corpus);
      const { good, total } = measure(net);
      expect(total).toEqual(256);
      expect(good).toEqual(203);
    });
  });

  describe('measure corpus', () => {
    test('It should measure an external corpus', () => {
      const net = new Neural();
      net.train(corpus);
      const { good, total } = net.measureCorpus(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(203);
    });
  });

  describe('run', () => {
    test('It should run an utterance in the model', () => {
      const net = train(corpus);
      const actual = run(net, 'Who are');
      expect(actual[0].intent).toEqual('smalltalk.acquaintance');
    });
    test('A list of valid potential intents can be provided', () => {
      const net = train(corpus);
      const actual = run(net, 'Who are', [
        'support.developers',
        'support.contact',
        'support.address',
      ]);
      expect(actual[0].intent).toEqual('support.developers');
    });
    test('If all perceptrons return an score of 0, then a None intent will be returned', () => {
      const net = train(corpus);
      const actual = run(net, 'Klatuu barada nikto');
      expect(actual[0].intent).toEqual('None');
    });
    test('It can use mono-intent detection', () => {
      const net = new Neural();
      net.train(corpus);
      const input = 'who are you, when is your birthday and who is your boss?';
      const actual = net.run(input);
      const expected = [
        {
          intent: 'smalltalk.boss',
          score: 0.7372071118768818,
        },
        {
          intent: 'smalltalk.birthday',
          score: 0.26274855823114524,
        },
        {
          intent: 'smalltalk.right',
          score: 0.000044329891972963164,
        },
      ];
      expect(actual).toEqual(expected);
    });
    test('It can use multi-intent detection', () => {
      const net = new Neural({ multi: true });
      net.train(corpus);
      const input = 'who are you, when is your birthday and who is your boss?';
      const actual = net.run(input);
      const expected = {
        monoIntent: [
          {
            intent: 'smalltalk.boss',
            score: 0.7372071118768818,
          },
          {
            intent: 'smalltalk.birthday',
            score: 0.26274855823114524,
          },
          {
            intent: 'smalltalk.right',
            score: 0.000044329891972963164,
          },
        ],
        multiIntent: [
          {
            tokens: ['who', 'are', 'you'],
            embeddings: [36, 64, 28],
            classifications: [
              {
                intent: 'smalltalk.acquaintance',
                score: 0.9986955806472481,
              },
              {
                intent: 'smalltalk.annoying',
                score: 0.0007621672107829141,
              },
              {
                intent: 'smalltalk.bad',
                score: 0.0005183030838014321,
              },
              {
                intent: 'smalltalk.hungry',
                score: 0.000023949058167544597,
              },
            ],
          },
          {
            tokens: ['when', 'is', 'your', 'birthday', 'and'],
            embeddings: [113, 8, 2, 114, 269],
            classifications: [
              {
                intent: 'smalltalk.birthday',
                score: 0.9891193351298617,
              },
              {
                intent: 'trivia.gc',
                score: 0.010880664870138371,
              },
            ],
          },
          {
            tokens: ['who', 'is', 'your', 'boss'],
            embeddings: [36, 8, 2, 141],
            classifications: [
              {
                intent: 'smalltalk.boss',
                score: 0.9999703942458513,
              },
              {
                intent: 'support.developers',
                score: 0.000029605754148707918,
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
      const net = new Neural();
      net.train(corpus);
      const json = net.toJSON();
      const net2 = new Neural();
      net2.fromJSON(json);
      const { good, total } = net2.measure(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(203);
    });
    test('It can be saved/loaded including perceptron changes', () => {
      const net = new Neural({ keepWeightsAndChanges: true });
      net.train(corpus);
      const json = net.toJSON({ saveChanges: true });
      const net2 = new Neural();
      expect(json.perceptrons[0].changes).toBeDefined();
      net2.fromJSON(json);
      const { good, total } = net2.measure(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(203);
    });
    test('It can be chosen to do not save the encoder', () => {
      const net = new Neural();
      net.train(corpus);
      const json = net.toJSON({ saveEncoder: false });
      const net2 = new Neural();
      net2.prepareCorpus(corpus.data);
      net2.fromJSON(json);
      const { good, total } = net2.measure(corpus);
      expect(total).toEqual(256);
      expect(good).toEqual(203);
    });
  });
});
