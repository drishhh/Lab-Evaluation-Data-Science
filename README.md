# Lab-Evaluation-Data-Science
Natural Language Inferencing (NLI) is a classic NLP (Natural Language Processing) problem that involves taking two sentences (the premise and the hypothesis ), and deciding how they are related- if the premise entails the hypothesis, contradicts it, or neither.

Here, we'll look at the Contradictory, My Dear Watson competition dataset, build a preliminary model using Tensorflow 2, Keras, and BERT, and prepare a submission file.

The training set contains a premise, a hypothesis, a label (0 = entailment, 1 = neutral, 2 = contradiction), and the language of the text. For more information about what these mean and how the data is structured, check out the data page: https://www.kaggle.com/c/contradictory-my-dear-watson/data

#FC to download the data outside of kaggle
#!kaggle competitions download -c contradictory-my-dear-watson 
We can use the pandas head() function to take a quick look at the training set.

Dataset from https://www.kaggle.com/hugoarmandopazvivas/contradictory-my-dear-watson-hapv

Preparing Data for Input
--The tutorial use a pre-trained BERT model combined with a classifier to make the predictions. In order to have better results than the Tutorial we will use a bigger model with the same architecture call RoBERTa (Robust optimization BERT), this model use basically 10 times more data and train more time than the original modelimage.png!

First, the input text data will be fed into the model (after they have been converted into tokens), which in turn outputs embbeddings of the words. The advantage to other types of embeddings is that the BERT embeddings (or RoBERTa embeddings because is the same architecture) are contextualized. Predictions with contextualized embeddings are more accurate than with non-contextualized embeddings.

![image](https://user-images.githubusercontent.com/96504008/158646868-97e7c382-4c86-4121-97a1-62369b9bce9b.png)

After we receive the embeddings for the words in our text from RoBERTa, we can input them into the classifier which will then in turn return the prediction labels 0,1, or 2.--

To start out, we can use a pretrained model. Here, we'll use a multilingual BERT model from huggingface. For more information about BERT, see: https://github.com/google-research/bert/blob/master/multilingual.md

First, we download the tokenizer.

-- We will first break down our text into tokens by using RoBERTa own tokenizer using AutoTokenizer and the model name in order for the tokenizer to know how to tokenize. This will download all the necessary files. This model includes more than 100 languages which is useful since our data also contains multiple languages.--

Creating & Training Model
Now, we can incorporate the BERT transformer into a Keras Functional Model. For more information about the Keras Functional API, see: https://www.tensorflow.org/guide/keras/functional.

This model was inspired by the model in this notebook: https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on-this-Competition, which is a wonderful introduction to NLP!

--Now, we are ready to build the actual model. As mentioned above, the final model will consist of a RoBERTa Large model that performs contextual embedding of the input token IDs which are then passed to a classifier that will return probabilites for each of the possible three labels "entailment" (0), "neutral" (1), or "contradiction" (2). The classifier consists of a regular densely-connected neural network.--

Creating & Training Model
Now, we can incorporate the BERT transformer into a Keras Functional Model. For more information about the Keras Functional API, see: https://www.tensorflow.org/guide/keras/functional.

This model was inspired by the model in this notebook: https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on-this-Competition, which is a wonderful introduction to NLP!

--Now, we are ready to build the actual model. As mentioned above, the final model will consist of a RoBERTa Large model that performs contextual embedding of the input token IDs which are then passed to a classifier that will return probabilites for each of the possible three labels "entailment" (0), "neutral" (1), or "contradiction" (2). The classifier consists of a regular densely-connected neural network.--

Creating & Training Model
Now, we can incorporate the BERT transformer into a Keras Functional Model. For more information about the Keras Functional API, see: https://www.tensorflow.org/guide/keras/functional.

This model was inspired by the model in this notebook: https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on-this-Competition, which is a wonderful introduction to NLP!

--Now, we are ready to build the actual model. As mentioned above, the final model will consist of a RoBERTa Large model that performs contextual embedding of the input token IDs which are then passed to a classifier that will return probabilites for each of the possible three labels "entailment" (0), "neutral" (1), or "contradiction" (2). The classifier consists of a regular densely-connected neural network.--
Creating & Training Model
Now, we can incorporate the BERT transformer into a Keras Functional Model. For more information about the Keras Functional API, see: https://www.tensorflow.org/guide/keras/functional.

This model was inspired by the model in this notebook: https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on-this-Competition, which is a wonderful introduction to NLP!

--Now, we are ready to build the actual model. As mentioned above, the final model will consist of a RoBERTa Large model that performs contextual embedding of the input token IDs which are then passed to a classifier that will return probabilites for each of the possible three labels "entailment" (0), "neutral" (1), or "contradiction" (2). The classifier consists of a regular densely-connected neural network.--


![image](https://user-images.githubusercontent.com/96504008/158647914-3d8a9943-a260-481b-924d-90476ad914cb.png)

--The graph above illustrates in a very detailed way what our model and its inputs look like: input_word_ids, input_mask, and input_type_ids are the 3 input variables for the BERT model, which in turn returns a tuple. The word embeddings that are stored in the first entry of the tuple are then given to the classifier which then returns 3 categorical probabilities. The question marks stand for the number of rows in the input data which are of course unknown.--


