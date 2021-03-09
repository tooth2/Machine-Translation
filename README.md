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
- 
## Approach 
1. preprocess the data by converting text to sequence of integers. 
    - tokenize function: returns tokenized input and the tokenized class.
    - pad function: returns padded input to the correct length.
2. build several deep learning models for translating the text into French. 
    - simple_model function: builds a basic RNN model
    - embed_model function: builds a RNN model using word embedding
        - The Embedding RNN is trained on the dataset. The Embedding RNN makes a prediction on the training dataset.
    - bd_model function: builds a bidirectional RNN model
        - The Bidirectional RNN is trained on the dataset. The Bidirectional RNN makes a prediction on the training dataset.
    - model_final function: builds and trains a model that incorporates embedding, and bidirectional RNN using the dataset.
3. run this models on English test to analyze their performance. 
