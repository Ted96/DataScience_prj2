import pandas as pd
from math import log
import numpy as np
import matplotlib.pyplot as plt
from math import log
from func import *


if __name__ == "__main__":

    train = pd.read_csv('train.tsv', sep='\t')

    plt.rcParams.update({'figure.max_open_warning': 0})

    goods = sum(train['Label'] == 1)
    bads = sum(train['Label'] == 2)
    pg = goods / (goods + bads)
    pb = bads / (goods + bads)

    H = -(pg * log(pg, 2) + pb * log(pb, 2))

    ### Changing all numerical Attributes into categorical ###
    convert_to_categorical(train)


    ###########################################
    ### PART1: Visualisation/Classification ###
    ###########################################


    ##########################################################
    ###                     Bar Charts                     ###
    ##########################################################

    category = [1, 2]

    attributes = ["Attribute1", "Attribute2q", "Attribute3", "Attribute4", "Attribute5q",
                  "Attribute6", "Attribute7", "Attribute8q", "Attribute9", "Attribute10",
                  "Attribute11q", "Attribute12", "Attribute13q", "Attribute14", "Attribute15",
                  "Attribute16q", "Attribute17", "Attribute18q", "Attribute19", "Attribute20"]

    normalize = 1  # flag

    goods = goods / 100.0
    bads = bads / 100.0

    normalizer = lambda x: ((bads - goods) * x + goods - 1) * normalize + 1

    our_metric = [0 for x in range(len(attributes))] # For testing purposes-it does not used
    info_gain = [0 for x in range(len(attributes))]

    # super loop
    k = 0
    for a in attributes:
        fig, ax = plt.subplots()
        attr = sorted(list(set(train[a])))
        p = [[0 for x in range(len(category))] for y in range(len(attr))]

        for i in range(len(category)):
            for j in range(len(attr)):
                p[j][i] = len(train[(train["Label"] == category[i]) & (train[a] == attr[j])]) / normalizer(i)

        ### Keep Information Gain of each Attribute (difference between good and bad for all categories)
        for j in range(len(attr)):                   # For testing purposes-it does not used
            our_metric[k] += abs(p[j][0] - p[j][1])

        info_gain[k] = information_gain(train, a, H)

        k += 1
        p_Good = [row[0] for row in p]
        p_Bad = [row[1] for row in p]

        index = np.arange(len(attr))
        bar_width = 0.25

        plt.bar(index, p_Good, bar_width, color='green', label='Good')
        plt.bar(index + bar_width, p_Bad, bar_width, color='red', label='Bad')

        if normalize == 1:
            plt.ylabel('% of all Good/Bad')
        else:
            plt.ylabel('# of applicants')
        plt.xticks(index, attr)
        plt.legend()

        plt.xlabel('Attributes')
        plt.title(a)
        plt.draw()  # temporally draw  & show all later.

    
    ##########################################################
    ###                     Box Plots                      ###
    ##########################################################

    numericals = ["Attribute2", "Attribute5", "Attribute13"]
    numer_description = ["Duration in months", "Credit amount", "Age in years"]


    i = 0
    for n in numericals:
        fig, ax = plt.subplots()
        train.boxplot(column=n, by="Label", ax=ax)
        ax.set_xticklabels(["Good", "Bad"])
        plt.ylabel(numer_description[i])
        fig.suptitle('')
        plt.title(numericals[i])
        plt.draw()
        i += 1
    plt.show()  # show all (blocks here if running from terminal)


    ##########################################################
    ###                 Classification                     ###
    ##########################################################

    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC                                # SVM
    from sklearn.ensemble import RandomForestClassifier        # Random Forest
    from sklearn.naive_bayes import GaussianNB , MultinomialNB # Naive Bayes

    ### Order Attributes (Features) by ascending Information Gain
    columns = ['metric', 'attribute']
    new_df = pd.DataFrame(columns=columns)
    new_df.metric = info_gain #our_metric info_gain
    new_df.attribute = attributes
    new_df = new_df.sort_values(by="metric")
    new_attributes = new_df.attribute

    ### Variables to be used
    attrs = 0
    categories = [1, 2]
    classifiers = [SVC(C=5, gamma=0.2, kernel='rbf'),  # SVM classifier (after calibration with GridSearchCV)
                   RandomForestClassifier(             # Random Forest Classifier (No of trees=No of attributes)
                       n_estimators=(len(attributes) - attrs)),
                   MultinomialNB()]                    # Naive Bayes Classifier
    clf = [[], [], []]
    mean_accuracy = [[0.0 for x in range(len(attributes))] for y in range(len(classifiers))]

    while attrs < len(attributes):  # Remove one feature each time (the one with the less information)

        ### Data to feed the classifiers (of descending number of attributes)
        df = train[new_attributes[attrs:len(attributes)]]
        df_dummy = pd.get_dummies(df)
        X_lsi = df_dummy.values

        ### 10-fold Cross Validation (Train Classifiers)
        for j in range(len(classifiers)):
            kf = KFold(n_splits=10, shuffle=True)
            fold = 0
            for train_index, test_index in kf.split(X_lsi):
                X_train_counts = np.array(X_lsi)[train_index]
                X_test_counts = np.array(X_lsi)[test_index]
                yTrue = np.array(train.Label)[test_index]

                clf[j] = classifiers[j].fit(X_train_counts, np.array(train.Label)[train_index])
                yPred = clf[j].predict((X_lsi)[test_index])

                accuracy = accuracy_score(yTrue, yPred, categories)
                mean_accuracy[j][attrs] += accuracy
                fold += 1
        attrs += 1

    ### Plot % Accuracy vs the # of used features (for all classifiers)
    xPoints = []
    for i in range(len(attributes)):
        x = len(attributes) - i
        xPoints.append(x)
    yPoints = [0.0 for y in range(len(attributes))]
    colors = ['blue', 'red', 'green']
    labels = ["SVM",
              "Rand Forest",
              "Naive Bayes"]

    for j in range(len(classifiers)):
        for i in range(len(attributes)):
            y = 10 * mean_accuracy[j][i]
            yPoints[i] = y
        plt.gca().invert_xaxis()
        plt.plot(xPoints, yPoints, color=colors[j], lw=1, label=labels[j])
        plt.xlim([20.5, 0.5])
        plt.ylim([60.0, 80.0])
        plt.xlabel('No of Features')
        plt.ylabel('% Accuracy')
        plt.title('Accuracy of Classifications')
        plt.legend(loc="upper right")
    plt.show()    # show accuracies plot (blocks here if running from terminal)


    ########################################################
    ### PART2: EvaluationMetric & Processing TestSet.csv ###
    ########################################################


    #########################################################
    ### Find best classifying algo && best no of features ###
    #########################################################
 
    best_Algorithm = np.argmax( [np.mean(mean_accuracy[i]) for i in range(len(classifiers))])
    best_No_of_features = \
        len(attributes) - np.argmax([ np.mean( mean_accuracy[best_Algorithm][i]) for i in range(len(attributes))])

    ### Data to feed the classifiers ###
    df = train[new_attributes[(len(attributes) - best_No_of_features):len(attributes)]]
    df_dummy = pd.get_dummies(df)
    X_lsi = df_dummy.values

    ### 10-fold Cross Validation (Train Classifiers) ###
    clf = [[], [], []]
    mean_acc = [0.0 for x in range(len(classifiers))]
    for j in range(len(classifiers)):
        kf = KFold(n_splits=10, shuffle=True)
        fold = 0
        for train_index, test_index in kf.split(X_lsi):
            X_train_counts = np.array(X_lsi)[train_index]
            X_test_counts = np.array(X_lsi)[test_index]
            yTrue = np.array(train.Label)[test_index]

            clf[j] = classifiers[j].fit(X_train_counts, np.array(train.Label)[train_index])
            yPred = clf[j].predict((X_lsi)[test_index])

            accuracy = accuracy_score(yTrue, yPred, categories)
            mean_acc[j] += accuracy
            fold += 1
        mean_acc[j] /= 10.0

    ### create dataframe with accuracy for all methods ###
    my_metrics = ["Accuracy"]
    out_dict = {
        "SVM": pd.Series(mean_acc[0], index=my_metrics),
        "Random Forest": pd.Series(mean_acc[1], index=my_metrics),
        "Naive Bayes": pd.Series(mean_acc[2], index=my_metrics)
    }

    out_df = pd.DataFrame(out_dict, index=my_metrics)
    out_df.to_csv(path_or_buf='EvaluationMetric_10fold.csv', index_label='Stat/Meas', sep='\t')

    ### print the created file ###
    out = pd.read_csv('EvaluationMetric_10fold.csv', sep='\t')
    print(out, end='\n\n')
    
    ### Classify the test file
    classify_test( "test.tsv",new_attributes, clf[best_Algorithm], best_No_of_features)
    
    ### print the created file ###
    print_test = pd.read_csv('testSet_Predictions.csv', sep='\t')
    print
    "\nPrinting the test file:"
    print(print_test)
    