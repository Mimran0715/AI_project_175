# CS175_Project
This directory contains the following Python Jupyter Notebooks:

**Project_Main.ipynb** - This notebook contains the entirety of our project. In it we explore the dataset, provide a bag of words encoding sample, obtain manual and pre-trained word embeddings for the Word2Vec and Stanford GLOVE embedding models, and feed these four embeddings through a simple convolutional neural network. We compare different values of hyperparameters including kernel size, dropout rate, and # of filter for the CNN and plot the corresponding model accuracies. We also provided some sample predictions on positive and negative movie reviews using the model. The following dependencies must be installed in order to run this code: tensorflow
* tensorflow-datasets
* numpy
* scipy
* sklearn
* nltk
* gensim
* matplotlib

**BagOfWords.ipynb** - This notebook contains a bag of words implementation of our dataset fed through a multilayer perceptron to compare with our word embeddings sentiment classifier. This implementation compares binary, tf-idf, frequency, and count vectors for the bag of words model. Sample model predictions on movie reviews are also provided. This code was collected from Jason Brownlee's tutorial, presented in the following link: https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/. This was added in to compare alongside the word embeddings implementation. The following dependencies must be installed to run this code: pandas, matplotlib, nltk, keras.

**BagOfWords_lite.ipynb** - This notebook contains a bag of words implementation with binary vectors. This can be run on the review_polarity dataset. The following dependencies must be installed to run this code: pandas, matplotlib, nltk, keras.

**ProjectMain_lite.ipynb** - This notebook contains an implementation of manually trained word embeddings fed through a simple convolutional neural network. This can be run on the review_polarity dataset. The following dependencies must be installed to run this code: tensorflow, tensorflow-datasets.

The review_polarity dataset used for this project is also provided. It contains 1000 positive movie reviews and 1000 negative movie reviews. 

In order to run this project, run the Project_Main_lite.ipynb and BagOfWords_lite.ipynb files. 

