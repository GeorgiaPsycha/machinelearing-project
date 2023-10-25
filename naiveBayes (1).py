#importing import-ant libraries
import pandas as pd

#--------------get_vectors--------------
#vectors = []
    #for row in dataset:
     #   vec = []
        #if the rating of the certain row is higher than 5 clf that as positive
      #  if int(row.split()[0]) > 5:
       #     vec.append(True)
        #else:
         #   vec.append(False)
        #checking if feature X exists on not in this review (to be existant there must be a substring of format : " X:")
        #for i in range(n, m):
            #if exists pass the value 1 in the list-vector
         #   if (" " + str(i) + ":") in row:
          #      vec.append(1)
           # else:
            #    vec.append(0)
        #vectors.append(vec)
    #return vectors

#description: manipulation of the data in the .feat files
#param @n, m int : n being the number of words to be skipped and m being the last word to be used as a feature from the voc
#      @dataset : the data we've load and want to turn into vectors
#      @key : the column of the dataset to be turned into vectors
#return vectors : a list of lists (each vector is represented with a list of 1,0s for the features
#and one True/False value for the class of the review)

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=4000)

word_index = tf.keras.datasets.imdb.get_word_index()
vocabulary = list()
for text in x_train:
  tokens = text.split()
  vocabulary.extend(tokens)

vocabulary = set(vocabulary)

def get_vectors(n, m, vocabulary, key):
    x_binary = []
    for text in tqdm(x_train):
        tokens = text.split()
        binary_vector = []
        if int(tokens[0]) > 5:
            binary_vector.append(True)
        else:
            binary_vector.append(False)
        for vocab_token in vocabulary:
            if vocab_token in tokens:
              binary_vector.append(1)
            else:
              binary_vector.append(0)
        x_binary.append(binary_vector)

    return x_binary

  
#--------------get_probOfFeat--------------
#description: calculates the probability of a given feature existing in pos/neg reviews
#param @feature : index of the word-feature in our vector
#      @vectors : list of our data vectors
#return tuple of 2 floats (feat_pos, feat_neg)
#feat_pos : the probability of our feature appearing in positive reviews (P(X=1|C=True/1))
#feat_neg : the probability of our feature appearing in negative reviews (P(X=1|C=False/0))
def get_probOfFeat(feature, vectors):
    #count var of how many reviews (pos/neg) the feature appears in
    sum_pos = 0
    sum_neg = 0
    #count var of how many pos or neg reviews there are
    pos = 0
    neg = 0
    for vec in vectors:
        if vec[0]:
            pos +=1
        else:
            neg +=1
        if vec[0] and (vec[feature] == 1):
            sum_pos+=1
        elif (vec[0] == False) and (vec[feature] == 1):
            sum_neg+=1
    #laplace 
    feat_pos = (sum_pos + 1) /(pos + 2)
    feat_neg = (sum_neg + 1) /(neg + 2)
    return feat_pos, feat_neg

#--------------get_probVoc--------------
#description: calculates the probabilities of all features in the vocab
#param @vectors : list of our data vectors
#return tuple of 2 lists (pos_prob, neg_prob)
#pos_prob : list of the probabilities of each feature to appear in pos reviews
#neg_prob : list of the probabilities of each feature to appear in neg reviews
def get_probVoc(vectors):
    pos_prob = []
    neg_prob = []
    #get the probabilities for each feature
    #skipping 1st index cause the class is located there
    for i in range(1, len(vectors[0])):
        pos_prob.append(get_probOfFeat(i, vectors)[0])
        neg_prob.append(get_probOfFeat(i, vectors)[1])
    return pos_prob, neg_prob

#--------------get_probClass--------------
#description: calculates the probability of a vector being pos or neg
#param @vectors : list of our data vectors
#return tuple of 2 floats (prob_pos, prob_neg)
#prob_pos : probability of a review being positive (C=True/1)
#prob_neg : probability of a review being negative (C=False/0)
def get_probClass(vectors):
    pos = 0
    neg = 0
    total = 0
    for vec in vectors:
        if (vec[0]):
            pos += 1
        else:
            neg += 1
        total +=1
    prob_pos = pos/total
    prob_neg = neg/total
    return prob_pos, prob_neg

#--------------train--------------
#description: trains the algorithm with provided inside train data
#param @n, m int : n being the number of words to be skipped and m being the last word to be used as a feature from the voc
#return : tuple of 4 floats (feat_probpos, feat_probneg, class_probpos, class_probneg)
#feat_probpros/neg : probabilities of our train data features appearing in pos/neg reviews
#class_probpos/neg : probabilities of our review belonging to the pos/neg ones
def train(n, m):
    #read the train data
    #reviews = pd.read_csv("neg", sep = "\n", header = None, names=['Review'])
    #make vectors out of the data
    vectors = get_vectors(n, m, vocabulary, 'Review')
    #remove a pair of commands (possitive/negative) and command before "getting the list..." comment
    #from comments to train the algorithm on different percentages of the train data
    #-----90% of vectors
    #positive_vectors = vectors[1125:11250]
    #negative_vectors = vectors[12375:22500]
    #-----80% of vectors
    #positive_vectors = vectors[2250:11250]
    #negative_vectors = vectors[13500:22500]
    #-----70% of vectors
    #positive_vectors = vectors[3375:11250]
    #negative_vectors = vectors[14625:22500]
    #-----60% of vectors
    #positive_vectors = vectors[4500:11250]
    #negative_vectors = vectors[15750:22500]
    #-----50% of vectors
    positive_vectors = vectors[5625:11250]
    negative_vectors = vectors[16875:22500]
    #-----40% of vectors
    #positive_vectors = vectors[6750:11250]
    #negative_vectors = vectors[18000:22500]
    #-----30% of vectors
    #positive_vectors = vectors[7875:11250]
    #negative_vectors = vectors[19125:22500]
    #-----20% of vectors
    #positive_vectors = vectors[9000:11250]
    #negative_vectors = vectors[20250:22500]
    #-----10% of vectors
    #positive_vectors = vectors[10125:11250]
    #negative_vectors = vectors[21375:22500]
    vectors = positive_vectors + negative_vectors
    #get the probability of the features (P(X|C))
    feat_probpos = get_probVoc(vectors)[0]
    feat_probneg = get_probVoc(vectors)[1]
    #get the probability of the class (P(C))
    class_probpos = get_probClass(vectors)[0]
    class_probneg = get_probClass(vectors)[1]
    return feat_probpos, feat_probneg, class_probpos, class_probneg

#--------------singleBayes--------------
#description: classifies a single vector given based on the set of probabilities train_probs
#param @n, m int : n being the number of words to be skipped and m being the last word to be used as a feature from the voc
#      @train_probs : probabilities needed from the training data
#      @vector : vector for his class to be predicted (positive/negative)
#return : boolean mul_totalpos > mul_totalneg (P(C=True/1|X) > P(C=False/0|X))
def singleBayes(train_probs, vector):
    #multiplication of probabilities of features in pos/neg reviews from our train data
    mul_totalpos = 1
    mul_totalneg = 1
    for i in range(1, len(vector)):
        if vector[i] == 1:
            mul_totalpos *= train_probs[0][i-1]
            mul_totalneg *= train_probs[1][i-1]
        else:
            mul_totalpos *= (1 - train_probs[0][i-1])
            mul_totalneg *= (1 - train_probs[1][i-1])
    mul_totalpos *= train_probs[2]
    mul_totalneg *= train_probs[3]
    return mul_totalpos > mul_totalneg#*0.1(threshold)

#--------------naiveBayes--------------
#description: trains our algorithm and then runs for a provided test dataset returning statistics of the classifications
#param @n, m int : n being the number of words to be skipped and m being the last word to be used as a feature from the voc
#return : tuple of 4 ints (TP, TN, FP, FN) -true positives/negatives, false positives/negatives-
def naiveBayes(n, m):
    print("Bernoulli Naive Bayes is running...")
    #train our algorithm and get the tuple of probability related data
    train_probs = train(n, m)
    print("Probabilities calculated - algorithm trained...")
    #read and create the test vectors
    test_df = pd.read_csv("labeledBow.feat", sep = "\n", header = None, names=['TReview'])
    test_vectors = get_vectors(n, m, test_df, 'TReview')
    print("Test data read and ready to be classified...")
    #accurate : counting the number of reviews that were categorized in pos/neg correctly
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for test in test_vectors:
        #classify each test using singleBayes
        judgement = singleBayes(train_probs, test)
        #if the classification was correct, increase the TP or TN
        if (judgement == test[0]):
            if(judgement):
                TP += 1
            else:
                TN += 1
        #if the classification was incorrect, increase the FP or FN
        else:
            if(judgement):
                FP +=1
            else:
                FN +=1
        #print ("Class of Vector was: ", test[0]," Naive Bayes Classifier predicted: ", judgement)
    #print stats
    accuracy = round((TP+TN)/(TP+TN+FP+FN), 4)
    precision = round(TP/(TP+FP), 4)
    recall =  round(TP/(TP+FN), 4)
    f1 = round((2*precision*recall)/(precision + recall), 4)
    message = "Bernoulli NaiveBayes run for:\n\tn: " + str(n) + "\n\tm: " + str(m)
    message += "\nResults: \n\tAccuracy: " + str(accuracy) + "\n\tPrecision: " + str(precision) + "\n\tRecall: " + str(recall) + "\n\tF1 score: " + str(f1)
    print(message)
    return TP, TN, FP, FN

naiveBayes(50, 2000)
