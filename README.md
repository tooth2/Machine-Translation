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
|Encoder/Decoder| | 63%/65%|
|final(Embedding+BidirectionalRNN+Encoder/Decoder)| | / | 

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
