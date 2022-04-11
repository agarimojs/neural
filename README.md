# @agarimo/neural
Neural Network for Conversational AI.

## Example of use

```javascript
const { Neural } = require('@agarimo/neural');
const corpus = require('./test/corpus-en.json');

const net = new Neural();
net.train(corpus);
const result = net.run('who are you?');
console.log(result);
```

It will return:
``` 
[
  { intent: 'smalltalk.acquaintance', score: 0.9994305864564091 },
  { intent: 'smalltalk.annoying', score: 0.00029939905400927795 },
  { intent: 'smalltalk.bad', score: 0.0002644902763842131 },
  { intent: 'smalltalk.hungry', score: 0.000005524213197446476 }
]
```

## Parameters that you can provide as settings

### log
If training must be logged. True means yes and will log in console, false means no log.
If you provide a function here, that mean yes, and this function will be used to log instead of the default funcition. This function will receive the status and the time.

Example log function:
```javascript
const defaultLogFn = (status, time) => console.log(`Epoch ${status.iterations} loss ${status.error} time ${time}ms`);
```

### encoder

This must be an instance of @agarimo/encoder.
By default, the encoder used only tokenizes, and works well for latin languages with separators.
If you have an stemmer or special text processing, consider build your own encoder and provide the instance to the Neural at the constructor.

### processor
If no encoder is provided, you can provide a processor for the encoder that will be created.
A processor is a funtion that accepts a text (string) and returns an array of tokens (array of strings). Example: "This is a test" can return something like ["this", "is", "a", "test"]
A processor can also be a NLP.js language item. Example of initialization with english NLP.js language:

```javascript
const { Neural } = require('@agarimo/neural');
const langEn = require('@nlpjs/lang-en');

const neural = new Neural({ processor: langEn });
```

### unknownIndex

Is how the unknown features will be represented. An unknown feature is a token (word) that is seen during processing time but was not there during the training, so the encoder is not able to calculate an embedding for it. 
Default is undefined, but you can provide other, example -1. Don't use positive numbers as these can be already be used by the encoder embeddings.

### useCache
Default is true. This will decide if the Neural instance uses an internal LRU cache for processing or not.

### cacheSize
Default is 1000. This is the size of the cache. As it's an LRU cache, will store only the last accessed items. Old items will be lost if the capacity of the cache is exceeded.

### iterations
Deafult is 20000. Number of maximum iterations to be used during training.

### errorThresh
Default is 0.00005. Error Threshold used as stop condition of the training. If the error becomes less than this number, then the training is stopped even if the maximum iterations are not reached.

### deltaErrorThres
Default is 0.000001. Delta error threshold used as stop condition of the training. The delta error threshold is the error variation between the current iteration and the previous one. If this variation is too small, means that the neural network is no longer converging. 

### learningRate
Default is 0.6. The learning rate.

### momentum
Default is 0.5. The momentum for the training, meaning how much inertia the current iteration has from the changes of the previous iteration.

### alpha
Default is 0.07. Alpha used for the activation and backpropagation functions. When a perceptron is run as the sum of bias and all inputs multiplied by it's weights, this sum is multiplied by the alpha before being returned.

### multi
Default is false. This is a beta feature for detecting multi intention in a sentence.
Example, given the sentence "who are you, when is your birthday and who is your boss?" with the test corpus provided at test/corpus-en.json, it will return an object like this:

```json
{                                                   
  "monoIntent": [                                   
    {                                               
      "intent": "smalltalk.boss",                   
      "score": 0.7464177512679069                   
    },                                              
    {                                               
      "intent": "smalltalk.birthday",               
      "score": 0.2535210691874059                   
    },                                              
    {                                               
      "intent": "smalltalk.right",                  
      "score": 0.00006117954468731187               
    }                                               
  ],                                                
  "multiIntent": [                                  
    {                                               
      "tokens": [                                   
        "who",                                      
        "are",                                      
        "you"                                       
      ],                                            
      "embeddings": [                               
        36,                                         
        64,                                         
        28                                          
      ],                                            
      "classifications": [                          
        {                                           
          "intent": "smalltalk.acquaintance",       
          "score": 0.9994305864564091               
        },                                          
        {                                           
          "intent": "smalltalk.annoying",           
          "score": 0.00029939905400927795           
        },                                          
        {                                           
          "intent": "smalltalk.bad",                
          "score": 0.0002644902763842131            
        },                                          
        {                                           
          "intent": "smalltalk.hungry",             
          "score": 0.000005524213197446476          
        }                                           
      ]                                             
    },                                              
    {                                               
      "tokens": [                                   
        "when",                                     
        "is",                                       
        "your",                                     
        "birthday",                                 
        "and"                                       
      ],                                            
      "embeddings": [                               
        113,                                        
        8,                                          
        2,                                          
        114,                                        
        269                                         
      ],                                            
      "classifications": [                          
        {                                           
          "intent": "smalltalk.birthday",           
          "score": 0.9874897374972665               
        },                                          
        {                                           
          "intent": "trivia.gc",                    
          "score": 0.01251026250273345              
        }                                           
      ]                                             
    },                                              
    {                                               
      "tokens": [                                   
        "who",                                      
        "is",                                       
        "your",                                     
        "boss"                                      
      ],                                            
      "embeddings": [                               
        36,                                         
        8,                                          
        2,                                          
        141                                         
      ],                                            
      "classifications": [                          
        {                                           
          "intent": "smalltalk.boss",               
          "score": 0.9999821681183945               
        },                                          
        {                                           
          "intent": "support.developers",           
          "score": 0.000017831881605606757          
        }                                           
      ]                                             
    }                                               
  ]                                                 
}                                                   
``` 

monoIntent is the usual answer of the network if is mono-intent.
multiIntent contains the three slices detected, with their classifications, tokens that compose this slice of the sentence and the embeddings of the tokens.