#importing import-ant libraries
import math
import pandas as pd

 
def get_vectors(n, m, dataset, key):
    vectors = []
    for row in dataset[key]:
        vec = {}
        #if the rating of the certain row is higher than 5 clf that as positive
        if int(row.split()[0]) > 5:
            vec["class"] = True
        else:
            vec["class"] = False
        #checking if feature X exists on not in this review (to be existant there must be a substring of format : " X:")
        for i in range(n, m):
            #if exists give the certain feature - key a value of 1
            if (" " + str(i) + ":") in row:
                vec[i+1] = 1
            else:
                vec[i+1] = 0
        vectors.append(vec)
    return vectors


#--------------ig_feature--------------
#description: calculates the information gain of a feature for a given list of vectors
#param @vectors : our data vectors
#      @feat : the feature to get the information gain (ig)
#      @entr_c : entropy of our class
#return : a float, the calculated information gain of our feature
def ig_feature(vectors,feat, entr_c):
    #cZxW count of instances where c = Z, x = W
    c1x1 = 0
    c1x0 = 0
    c0x0 = 0
    c0x1 = 0
    x1 = 0
    x0 = 0
    for vec in vectors: 
        if (vec.get("class") and vec.get(feat) == 1):
            c1x1 += 1
            x1 += 1
        elif (vec.get("class") and vec.get(feat) == 0):
            c1x0 += 1
            x0 += 1
        elif ((not vec.get("class")) and vec.get(feat) == 1):
            c0x1 += 1
            x1 += 1
        elif ((not vec.get("class")) and vec.get(feat) == 0):
            c0x0 += 1
            x0 += 1
    #probabilities initialized to 1 to avoid any mathematical error (division by zero, log(0))
    probc1x1 = 1
    probc0x1 = 1
    probc1x0 = 1
    probc0x0 = 1  
    #if the counts for the each desired probability
    #are both larger than 0, calculate the probability
    if (c1x1 != 0 and x1 != 0):
        probc1x1 = c1x1 / x1
    if (c0x1 != 0 and x1 != 0):
        probc0x1 = c0x1 / x1
    if (c1x0 != 0 and x0 != 0):
        probc1x0 = c1x0 / x0
    if (c0x0 != 0 and x0 != 0):
        probc0x0 = c0x0 / x0      
    #calculate the entropy of C given X, for both possible values of x (0, 1)
    hcx1 = - (probc1x1 * math.log(probc1x1, 2)) - (probc0x1 * math.log(probc0x1, 2))
    hcx0 = - (probc1x0 * math.log(probc1x0, 2)) - (probc0x0 * math.log(probc0x0, 2))
    #calculate probablity of x = 1, 0
    px1 = x1 / len(vectors)
    px0 = x0 / len(vectors)
    #calculate entropy of given feature using the already calculated 
    return (entr_c - (px1 * hcx1) - (px0 * hcx0))

#--------------infoG--------------
#description: calculates the ig for all features given a list of vectors and features
#param @vectors : our data vectors
#      @features : list of features to calculate information gain for
#return : ig a dictionary where each key is the position (line) of a word- feature
#            in the voc file and each value is the information gain of this feature
#            calculated using ig_feature
def infoG(vectors, features):
    #get the entropy of the class
    entr_c = entr_class(vectors)
    ig = {}
    for f in features:
        ig[f] = ig_feature(vectors, f, entr_c)
    return ig

#--------------entr_class--------------
#description: calculates the entopy of our class (negative/positive)
#param @vectors : our data vectors
#return : a float which is the calculated entropy of our class
def entr_class(vectors):
    pos = 0
    neg = 0
    for vec in vectors:
        if (vec.get("class")):
            pos += 1
        else:
            neg += 1
    prob_pos = pos / len(vectors)
    prob_neg = neg / len(vectors)
    if (prob_neg == 1 or prob_pos == 1):
        return 0
    return (- (prob_pos * math.log(prob_pos, 2)) - (prob_neg * math.log(prob_neg, 2)))

#--------------get_percClass--------------
#description: calculates percentage of each class in a given set of vectors
#param @vectors : our data vectors
#return : a tuple of 2 floats (perc_pos, perc_neg)
#         perc_pos : percentage of positive reviews in our vectors
#         perc_neg : percetange of negative reviews in our vectors
def get_percClass(vectors):
    pos = 0
    neg = 0
    total = 0
    for vec in vectors:
        if (vec.get("class")):
            pos += 1
        else:
            neg += 1
        total +=1
    perc_pos = pos/total
    perc_neg = neg/total
    return perc_pos, perc_neg

#--------------make_tree--------------
#description: makes a decision tree using the id3 algorithm
#param @vectors : our data vectors
#      @best_feat : the best feature, aka the feature with the highest information gain to divide the vectors
#      @features : list of features left
#      @perc_class : tuple of two floats, one the "default" percentage of positive reviews, other of negative
#      @dep : the maximum depth we want our tree to reach
#return : tree a list of size 3, that contains:
#                0. the best feature, aka the one that's got the max information gain
#                1. the left subtree of our tree (calculated for all the vectors that have the previously mentioned feature)
#                2. the right subtree of our tree (calculated for all the vectors that don't have the previously mentioned feature)
#         due to the recursive calls of the function our "leaf" values (last children - subtrees) will be either true or false
def make_tree(vectors, best_feat, features, perc_class, dep):
    #if maximum depth reached, return the class of the previous call ("default class")
    if (dep == 0):
        return perc_class[0] > perc_class[1] #have threshold in the inequality instead of percentage of other class
    #if not any other examples left, return the class of the previous call ("default class")
    elif (len(vectors) == 0):
        return perc_class[0] > perc_class[1] #have threshold in the inequality instead of percentage of other class
    #if our examples all belong to one class return the class
    #if not any other features left in our vectors, return the most common class of the current vectors-examples
    elif (len(features) == 0):
        current_percClass = percentageofClass(vectors)
        return current_percClass[0] > current_percClass[1] #have threshold in the inequality instead of percentage of other class
    else:
        #form a new tree-subtree
        tree = []
        #get the information gain for the features in the current vectors
        information = informationGain(vectors, features)
        #get the best feature out of the information gain
        best_feat = max(information, key = information.get)
        perc_class =percentageofClass(vectors)
        #append - label the current node with the best feature
        tree.append(best_feat)
        #create 2 new lists : hasfeat, nofeat to split our vectors
        #             hasfeat : vectors that have the current feature will be appended here
        #             nofeat :  vectors that do not have the current feature will be appended here
        hasfeat = []
        nofeat = []
        #spliting the vectors
        for vec in vectors:
            if (vec.get(best_feat) == 1):
                hasfeat.append(vec)
            else:
                nofeat.append(vec)
        #removing the current best feature from the available features 
        features.remove(best_feat)
        #make two 2 subtrees - children of our starting node, hasSubtree, nSubtree
        #           hasSubtree : will create a new tree based on the examples that had the current feature
        #           noSubtree : will respectively create a new tree based on the examples that had not the current feature
        hasSubtree = make_tree(hasfeat, best_feat, features, perc_class, dep -1)
        #adding the children to our tree
        tree.append(hasSubtree)
        noSubtree = make_tree(nofeat, best_feat, features, perc_class, dep -1)
        tree.append(noSubtree)
        return tree

#--------------traverse--------------
#description: traverses a given decision tree for a given vector
#param @node : the tree-node we "look at" while traversing
#      @vector : the vector that will be used to traverse the tree
#return : boolean True or False, the value found at the leaf when finishing the traversing
def traverse(node, vector):
    #if leaf reached : leaves being boolean values, return that value
    if (node== False or node == True):
        return node
    else:
        #get the feature we have to check now
        value = vector.get(node[0])
        #if feature "exists" in our vector, traverse the left subtree found at node[1]
        if (value == 1):
            return traverse(node[1], vector)
        #if feature doesn't exist, traverse right subtree at node[2]
        else:
            return traverse(node[2], vector)

#--------------train--------------
#description: trains the algorithm with provided inside train data
#param @n, m int : n being the number of words to be skipped and m being the last word to be used as a feature from the voc
#return : decision tree
def train(n, m):
    #read the train data
    reviews = pd.read_csv("data/labeledBow.feat", sep = "\n", header = None, names=['Review'])
    #make vectors out of the data
    vectors = get_vectors(n, m, reviews, 'Review')
    #remove a pair of commands (possitive/negative) and command before "getting the list..." comment
    #from comments to train the algorithm on different percentages of the train data
    positive_vectors = vectors[5625:11250]
    negative_vectors = vectors[16875:22500]
    vectors = positive_vectors + negative_vectors
    #getting the list of our attributes and removing the class, since we don't need it
    features = list(vectors[0].keys())
    features.remove("class")
    #get the information gain for the features in the current vectors
    information = infoG(vectors, features)
    #get the best feature out of the information gain
    best_feat = max(information, key = information.get)
    #get percectages of classes for current vectors - examples
    current_percClass = get_percClass(vectors)
    #make the tree based on our training data - vectors
    tree = make_tree(vectors, best_feat, features, current_percClass, 15)
    return tree
        
#--------------ID3--------------
#description: trains our algorithm and then runs for a provided test dataset returning statistics of the classifications
#param @n, m int : n being the number of words to be skipped and m being the last word to be used as a feature from the voc
#return : tuple of 4 ints (TP, TN, FP, FN) -true positives/negatives, false positives/negatives-
def id3(n, m):
    print("ID3 is running...")
    #train our algorithm and get the decision tree
    tree = train(n, m)
    print("Tree made - algorithm trained...")
    #read and create the test vectors
    test_df = pd.read_csv("data/labeledBow.feat", sep = "\n", header = None, names=['TestReview'])
    test_vectors = get_vectors(n, m, test_df, 'TestReview')
    print("Test data read and ready to be classified...")
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for test in test_vectors:
        #for each test traverse the decision tree and get the value found at our target leaf
        judgement = traverse(tree, test)
        #if the classification was correct, increase the TP or TN
        if (test.get("class") == judgement):
            if(judgement):
                TP += 1
            else:
                TN += 1
        #if the classification was incorrect, increase the FP or FN
        else:
            if(judgement):
                FP += 1
            else:
                FN += 1
        #print("Class of Vector was: ", test.get("class")," ID3 Classifier predicted: ", judgement)
    #print stats
    accuracy = round((TP+TN)/(TP+TN+FP+FN), 4)
    precision = round(TP/(TP+FP), 4)
    recall =  round(TP/(TP+FN), 4)
    f1 = round((2*precision*recall)/(precision + recall), 4)
    message = "ID3 run for:\n\tn: " + str(n) + "\n\tm: " + str(m) +" "
    message += "\nResults: \n\tAccuracy: " + str(accuracy) + "\n\tPrecision: " + str(precision) + "\n\tRecall: " + str(recall) + "\n\tF1 score: " + str(f1)
    print(message)
    return TP, TN, FP, FN

id3(1, 100)
