from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import linalg
import math

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self.label = label
        
    def __str__(self):
        return str(self.label)

class FeatureVector:
    def __init__(self):
        self._fv = {}
        
    def add(self, index, value):
        self._fv[index] = value
        
    def get(self, index):
        return self._fv[index]

    def keys(self):
        return self._fv.keys()

class Instance:
    def __init__(self, feature_vector, label):
        self.feature_vector = feature_vector
        self.label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

# Specific predictor class with implemented perceptron algorithms
class PerceptronPredictor:
    def __init__(self, learn_rate, iterations, lamb, maxfeature):
        self.weight = {}
        self.weight_nonsparse = np.zeros(maxfeature + 1)
        self.use_nonsparse = 0
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.lamb = lamb
        self.weight_sum = {} #only used for averaged_perceptron

    #returns true if the Perceptron (using current weight vector) correctly predicts the label
    #false if otherwise
    def predict(self, instance):
        label = instance.label
        fv = instance.feature_vector

        if self.use_nonsparse:
            # initialize the sum for the dot_product
            dot_product = 0
            for v in fv.keys():
                dot_product += self.weight_nonsparse[v] * fv.get(v)

            # if the dot product sum is >= 0, then the predict_label = 1; otherwise, 0
            if dot_product >= 0:
                return 1
            else:
                return 0
        else:
            #initialize the sum for the dot_product
            dot_product = 0

            #iterate according to the non-zero values in the weight vector.
            #all other values are zero, so the dot product does not count them.
            for v in fv.keys():
                #Look for w in the feature_vector to see if it exists, if so then add to dot product
                try:
                    dot_product += self.weight[v] * fv.get(v)
                except KeyError:
                    pass

            #if the dot product sum is >= 0, then the predict_label = 1; otherwise, 0
            if dot_product >= 0:
                return 1
            else:
                return 0



    def train(self, instances, algorithm):

        if (algorithm == "margin_perceptron"):
            for k in range(0, self.iterations):
                for instance in instances:
                    #get information from the instance
                    label = instance.label
                    fv = instance.feature_vector

                    # initialize the sum for the dot_product
                    dot_product = 0

                    # iterate according to the non-zero values in the weight vector.
                    # all other values are zero, so the dot product does not count them.
                    for v in fv.keys():
                        # Look for w in the feature_vector to see if it exists, if so then add to dot product
                        try:
                            dot_product += self.weight[v] * fv.get(v)
                        except KeyError:
                            pass

                    ylabel = label.label #extract the integer label from the ClassificationLabel
                    if ylabel == 0:
                        y = -1
                    else:
                        y = 1
                    #margin-condition
                    if (y * dot_product < 1):
                        #for every non-zero key value in the feature vector
                        for v in fv.keys():
                            try:
                                self.weight[v] += self.learn_rate * y * fv.get(v)
                            except KeyError:
                                self.weight[v] = self.learn_rate * y * fv.get(v)

        elif (algorithm == "pegasos"):
            self.use_nonsparse = 1
            time = 1
            for k in range(0, self.iterations):
                for instance in instances:
                    # get information from the instance
                    ylabel = instance.label.label
                    fv = instance.feature_vector

                    if ylabel == 0:
                        y = -1
                    else:
                        y = 1

                    # Get the dot product of w and x
                    dot_product = 0
                    for v in fv.keys():
                        dot_product += self.weight_nonsparse[v] * fv.get(v)

                    #define indicator function
                    def indicator(y, dot_product):
                        if y*dot_product < 1:
                            return 1
                        else:
                            return 0

                    #update weight vectors
                    for index in range(0, len(self.weight_nonsparse)):
                        try:
                            self.weight_nonsparse[index] = (1.0 - 1.0/time)*self.weight_nonsparse[index] + (1.0/(self.lamb*time))*indicator(y, dot_product)*y*fv.get(index)
                        except KeyError:
                            self.weight_nonsparse[index] = (1.0 - 1.0/time)*self.weight_nonsparse[index]

                    time += 1

        # OLD FROM HW1 - even though code is repetitive, wanted to isolate it from HW2
        else:
            #train according to the number of iterations given
            for k in range(0, self.iterations):

                #for each iteration of training, loop over each instance
                for instance in instances:
                    #get information from the instance
                    label = instance.label
                    fv = instance.feature_vector

                    # predict using the current weight vector
                    yhat = self.predict(instance)
                    ylabel = label.label #extract the integer label from the ClassificationLabel

                    #convert the label given to {-1, 1} for the perceptron algorithm
                    if ylabel == 0:
                        y = -1
                    else:
                        y = 1

                    #compare perceptron prediction to actual ylabel
                    if ylabel == yhat:
                        continue
                    #if perceptron is wrong, update the weight vector
                    else:
                        #for every non-zero key value in the feature vector
                        for v in fv.keys():
                            try:
                                self.weight[v] += self.learn_rate * y * fv.get(v)
                            except KeyError:
                                self.weight[v] = self.learn_rate * y * fv.get(v)

                    #if averaged_perceptron is selected, update accordingly
                    if algorithm == "averaged_perceptron":
                        #add all the non-zero weight vector values to the cumulating weight_sum vector
                        for w in self.weight.keys():
                            try:
                                self.weight_sum[w] += self.weight[w]
                            except KeyError:
                                self.weight_sum[w] = self.weight[w]

            #at the end of all the training iterations, update the weight vector as the averaged of all training iterations
            if algorithm == "averaged_perceptron":
                self.weight = self.weight_sum

class knnPredictor:
    def __init__(self, k, max_feature):
        self.k = k

    def predict(self, instance):
        #distance list is a list of tuples(distance, label)
        distance_list = []

        for other_instance in self.instance_list:
            distance = 0

            # get combined key list for the distance formula
            #keyslist = list(set(instance.feature_vector.keys() + other_instance.feature_vector.keys()))
            keyslist = other_instance.feature_vector.keys()
            for key in keyslist:
                try:
                    distance += math.pow(instance.feature_vector.get(key) - other_instance.feature_vector.get(key), 2)
                except KeyError:
                    distance += math.pow(other_instance.feature_vector.get(key), 2)

            distance = math.sqrt(distance)
            distance_list.append((distance, other_instance.label.label))

        #sort list to get nearest neighbors
        sorted_list = sorted(distance_list, key = lambda distance: distance[0])

        label_list = {}
        #now predict label using k nearest neighbors
        for i in range(0, self.k):
            try:
                label_list[sorted_list[i][1]] += 1
            except KeyError:
                label_list[sorted_list[i][1]] = 1

        max_count = 0
        maxlabel = 999

        print label_list
        for label in label_list.keys():
            if label_list[label] > max_count:
                max_count = label_list[label]
                maxlabel = label
            elif label_list[label] == max_count:
                if label < maxlabel:
                    maxlabel = label

        print maxlabel
        return maxlabel


    def train(self, instances, algorithm):
        self.instance_list = instances

class BoostPredictor:
    def __init__(self, iterations):
        self.iterations = iterations
        self.a = {}
        self.opt_hypos = []

    def predict(self, instance):
        label_list = {}

        for t in range(0, len(self.opt_hypos)):
            hypothesis_t = self.opt_hypos[t]
            try:
                feature_val = instance.feature_vector.get(hypothesis_t["j"])
            except KeyError:
                feature_val = 0

            if feature_val <= hypothesis_t["cutoff"]:
                label = hypothesis_t["below_label"]
            else:
                label = hypothesis_t["above_label"]

            print label
            print "j, c, below, above:", hypothesis_t["j"], hypothesis_t["cutoff"], hypothesis_t["below_label"], \
            hypothesis_t["above_label"]

            try:
                label_list[label] += self.a[t]
            except KeyError:
                label_list[label] = self.a[t]

        print "label list", label_list
        max_count = 0
        for label in label_list.keys():
            if label_list[label] > max_count:
                max_count = label_list[label]
                maxlabel = label

        print "final label", maxlabel
        print "actual label", instance.label.label
        if maxlabel == -1:
            return 0
        else:
            return maxlabel

    def train(self, instances, algorithm):

        # define helper functions
        def frequentLabel(label_list):
            max_count = 0
            for label in label_list.keys():
                if label_list[label] > max_count:
                    max_count = label_list[label]
                    maxlabel = label
            return maxlabel

        def changeLabel(label):
            if label == 0:
                return -1
            else:
                return label

        n = len(instances)

        #initialize distribution
        self.distribution = np.multiply((1.0/n), np.ones(n))

        # structure= {feature: [(feature value, label), ..... i instances]}
        feature_list = {}

        # structure = [(j, c, [label, label])]
        hypotheses = []

        #get a list of all the feature numbers available
        keyList = []
        for instance in instances:
            for key in instance.feature_vector.keys():
                if key not in keyList:
                    keyList.append(key)

        instance_no = 0
        #loop through all instances
        for instance in instances:

            for key in keyList:
                #try getting the feature value from the instance.
                try:
                    feature_val = instance.feature_vector.get(key)
                #if it doesn't exist, that means the key was in the feature_list, so we put 0
                except KeyError:
                    feature_val = 0

                #now try adding the instance to the feature list under the feature number
                try:
                    feature_list[key].append((feature_val, changeLabel(instance.label.label), instance_no))
                #if this is a new feature number, add a new list under feature_list
                except KeyError:
                    feature_list[key] = [(feature_val, changeLabel(instance.label.label), instance_no)]

            instance_no += 1

        print "instances loaded"
        for j in feature_list.keys():
            #sort the list based on feature magnitude
            sorted_feature = sorted(feature_list[j], key = lambda feature: feature[0])

            print "feature", j
            #try n - 1 number of c cutoffs
            for c in range(1, len(sorted_feature)):
                below_label_list = {}
                #compute dominant label everything below
                for m in range(0, c):
                    try:
                        below_label_list[sorted_feature[m][1]] += 1
                    except KeyError:
                        below_label_list[sorted_feature[m][1]] = 1
                belowlabel = frequentLabel(below_label_list)

                #compute dominant label everything above
                above_label_list = {}
                for m in range(c, len(sorted_feature)):
                    try:
                        above_label_list[sorted_feature[m][1]] += 1
                    except KeyError:
                        above_label_list[sorted_feature[m][1]] = 1
                abovelabel = frequentLabel(above_label_list)

                #each hypothesis is a list of length n (instances)
                #that contains a tuple of the predicted label and the correct label
                hypo = []

                for m in range(0, len(sorted_feature)):
                    if m < c:
                        hypo.append((belowlabel, sorted_feature[m][2]))
                    else:
                        hypo.append((abovelabel, sorted_feature[m][2]))

                hypo = sorted(hypo, key = lambda feature: feature[1])

                cutoff = 0.5*(sorted_feature[c][0] + sorted_feature[c-1][0])

                hypotheses.append({"j":j, "cutoff": cutoff, "below_label": belowlabel, "above_label": abovelabel, "h(x)": hypo})

        for t in range(0, self.iterations):
            #find ht optimal classifier
            min_err = 999999999
            for hypothesis in hypotheses:
                h = hypothesis["h(x)"]
                err = 0
                for i in range(0, len(h)):
                    if h[i][0] != changeLabel(instances[i].label.label):
                        err += self.distribution[i]

                if err < min_err:
                    min_err = err
                    hypothesis_t = hypothesis
                    ht = h

            #calculate alpha
            if min_err < 0.000001:
                break
            else:
                self.a[t] = 0.5*math.log((1 - min_err)/min_err)

            #update distribution
            dist_sum = 0

            for i in range(0, n):
                dist_sum += self.distribution[i]*math.exp(-self.a[t]*changeLabel(instances[i].label.label)*ht[i][0])

            for i in range(0, n):
                self.distribution[i] = self.distribution[i]*math.exp(-self.a[t]*changeLabel(instances[i].label.label)*ht[i][0])/dist_sum

            self.opt_hypos.append(hypothesis_t)

            print min_err



