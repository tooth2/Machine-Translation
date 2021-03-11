# Machine-Translation
- A deep neural network that functions as part of an end-to-end machine translation pipeline.
- This pipeline will accept English text as input and return the French translation. 
- Implementation will explore several recurrent neural network (RNN) architectures and compare their performance.

## Background
1. Feature Extraction and Embeddings
    - extract features from text
    - embedding algorithms:  Word2Vec and Glove
2. Modeling
    - Deep learning models on machine translation, topic models, and sentiment analysis
3. Deep Learning Attention
    - attention, deep learning method empowering applications like Google Translate.
    - additive and multiplicative attention in applications like machine translation, text summarization, and image captioning.
    - TheTransformer that extend the use of attention to eliminate the need for RNNs.

## Approach 
1. preprocess the data by converting text to sequence of integers. 
    - tokenize function: returns tokenized input and the tokenized class.
    - pad function: returns padded input to the correct length.
    - Max English sentence length: 15 vs. Max French sentence length: 21
    - English vocabulary size: 199 vs. French vocabulary size: 344
2. build several deep learning models for translating the text into French. 
    - simple_model function: builds a basic RNN model
    - embed_model function: builds a RNN model using word embedding
        - The Embedding RNN is trained on the dataset. The Embedding RNN makes a prediction on the training dataset.
    - bd_model function: builds a bidirectional RNN model
        - The Bidirectional RNN is trained on the dataset. The Bidirectional RNN makes a prediction on the training dataset.
    - encoder + decoder model function: Encoder(RNN)-->(hidden state- Tensors-LSTM)-->Decoder(RNN)
    - model_final function: builds and trains a model that incorporates embedding, encoder/decoder and bidirectional RNN using the dataset.
3. run this models on English test to analyze their performance. 

## Result 
| Model | Network architecture | Train/Validation Accuracy|
|--|--|--|
|Simple basic RNN Model |<img src="/images/rnn.png" width="400"/>|67%/69% | 
|RNN with Embedding|<img src="/images/embedding.png" width="400" /> | 90%/92%|
|Bidirectional RNN |<img src="/images/bidirectional.png" width="400" /> |67%/69% |
|Encoder/Decoder|  | 63%/65%|
|final(Embedding+BidirectionalRNN+Encoder/Decoder)|Final Model | 93%/95% | 

```Encode/Decoder Model 
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_4 (InputLayer)             (None, 21, 1)         0                                            
____________________________________________________________________________________________________
lstm_7 (LSTM)                    [(None, 400), (None,  643200      input_4[0][0]                    
____________________________________________________________________________________________________
repeat_vector_4 (RepeatVector)   (None, 21, 400)       0           lstm_7[0][0]                     
____________________________________________________________________________________________________
lstm_8 (LSTM)                    (None, 21, 400)       1281600     repeat_vector_4[0][0]            
                                                                   lstm_7[0][1]                     
                                                                   lstm_7[0][2]                     
____________________________________________________________________________________________________
time_distributed_32 (TimeDistrib (None, 21, 345)       138345      lstm_8[0][0]                     
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 21, 345)       0           time_distributed_32[0][0]        
====================================================================================================
```

```Final Model 
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_8 (InputLayer)             (None, 21)            0                                            
____________________________________________________________________________________________________
embedding_8 (Embedding)          (None, 21, 256)       51200       input_8[0][0]                    
____________________________________________________________________________________________________
lstm_16 (LSTM)                   [(None, 256), (None,  525312      embedding_8[0][0]                
____________________________________________________________________________________________________
lstm_17 (LSTM)                   [(None, 256), (None,  525312      embedding_8[0][0]                
____________________________________________________________________________________________________
concatenate_9 (Concatenate)      (None, 512)           0           lstm_16[0][0]                    
                                                                   lstm_17[0][0]                    
____________________________________________________________________________________________________
repeat_vector_7 (RepeatVector)   (None, 21, 512)       0           concatenate_9[0][0]              
____________________________________________________________________________________________________
concatenate_7 (Concatenate)      (None, 512)           0           lstm_16[0][1]                    
                                                                   lstm_17[0][1]                    
____________________________________________________________________________________________________
concatenate_8 (Concatenate)      (None, 512)           0           lstm_16[0][2]                    
                                                                   lstm_17[0][2]                    
____________________________________________________________________________________________________
lstm_18 (LSTM)                   (None, 21, 512)       2099200     repeat_vector_7[0][0]            
                                                                   concatenate_7[0][0]              
                                                                   concatenate_8[0][0]              
____________________________________________________________________________________________________
time_distributed_35 (TimeDistrib (None, 21, 345)       176985      lstm_18[0][0]                    
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 21, 345)       0           time_distributed_35[0][0]        
====================================================================================================
Total params: 3,378,009
Trainable params: 3,378,009
Non-trainable params: 0
___________________________
```
## Reference 
- [Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)
- [Seq2Seq model](http://jalammar.github.io/)
- [Neural Machine Translation](https://arxiv.org/abs/1409.0473)
- [Attention based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Google's Neural Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)
- [Text Summarization with Seq2Seq RNN](https://arxiv.org/pdf/1602.06023.pdf)
- [Attention is All you need](https://arxiv.org/abs/1706.03762)
- [Machine Translation Dataset](http://www.statmt.org/)
- [Learning Enigma with RNN](https://greydanus.github.io/2017/01/07/enigma-rnn/)
