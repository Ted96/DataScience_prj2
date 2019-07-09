import pandas as pd


def classify_test(test_file ,new_attributes , clf , best_No_of_features):

    print("classifying",best_No_of_features, "feat. with", clf)

    ### Open test file ###
    test = pd.read_csv(test_file, sep='\t')

    ### Changing all numerical Attributes into categorical ###
    convert_to_categorical(test)

    ### Data to feed the classifiers ###
    df_test = test[new_attributes[(20 - best_No_of_features):20]]
    df_dummy_test = pd.get_dummies(df_test)
    X_lsi_test = df_dummy_test.values

    predicted = clf.predict(X_lsi_test)
    characterise = ["" for x in range(len(predicted))]
    for i in range(len(predicted)):
        if predicted[i] == 1:
            characterise[i] = "Good"
        else:
            characterise[i] = "Bad"

    ### create dataframe with all the metrics ###
    out_test = {
        "Client_ID": pd.Series(test.Id),
        "Predicted_Category": pd.Series(characterise)
    }
    out_df_test = pd.DataFrame(out_test)
    out_df_test.to_csv(path_or_buf='testSet_Predictions.csv', index=False, sep='\t')



def information_gain(dataset , attribute, H):

    def E(x, y):

        from math import log

        if x * y == 0:
            e = 0
        else:
            e = - ((x / (x + y)) * log((x / (x + y)), 2) + (y / (x + y)) * log((y / (x + y)), 2))

        return e

    h=0
    total = len(dataset)
    values = list(set(dataset[attribute]))

    for v in values:
        g = len(dataset[(dataset["Label"] == 1) & (dataset[attribute] == v)])
        b = len(dataset[(dataset['Label'] == 2) & (dataset[attribute] == v)])

        h += E(g,b) * ( g + b)

    #print("en of", attribute," vs=",values[:3] , "..  h=", H- h/total )
    return H - h/total



def convert_to_categorical(dataset):

    my_Attribute = ["" for x in range(len(dataset.Id))]

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute2[i] <= 9:
            my_Attribute[i] = "A21"
        elif dataset.Attribute2[i] <= 12:
            my_Attribute[i] = "A22"
        elif dataset.Attribute2[i] <= 18:
            my_Attribute[i] = "A23"
        elif dataset.Attribute2[i] <= 24:
            my_Attribute[i] = "A24"
        else:
            my_Attribute[i] = "A25"
    dataset["Attribute2q"] = my_Attribute

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute5[i] <= 1000:
            my_Attribute[i] = "A51"
        elif dataset.Attribute5[i] <= 1500:
            my_Attribute[i] = "A52"
        elif dataset.Attribute5[i] <= 2500:
            my_Attribute[i] = "A53"
        elif dataset.Attribute5[i] <= 4000:
            my_Attribute[i] = "A54"
        else:
            my_Attribute[i] = "A55"
    dataset["Attribute5q"] = my_Attribute

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute8[i] == 1:
            my_Attribute[i] = "A81"
        elif dataset.Attribute8[i] == 2:
            my_Attribute[i] = "A82"
        elif dataset.Attribute8[i] == 3:
            my_Attribute[i] = "A83"
        elif dataset.Attribute8[i] == 4:
            my_Attribute[i] = "A84"
    dataset["Attribute8q"] = my_Attribute

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute11[i] == 1:
            my_Attribute[i] = "A111"
        elif dataset.Attribute11[i] == 2:
            my_Attribute[i] = "A112"
        elif dataset.Attribute11[i] == 3:
            my_Attribute[i] = "A113"
        elif dataset.Attribute11[i] == 4:
            my_Attribute[i] = "A114"
    dataset["Attribute11q"] = my_Attribute

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute13[i] <= 25:
            my_Attribute[i] = "A131"
        elif dataset.Attribute13[i] <= 30:
            my_Attribute[i] = "A132"
        elif dataset.Attribute13[i] <= 35:
            my_Attribute[i] = "A133"
        elif dataset.Attribute13[i] <= 45:
            my_Attribute[i] = "A134"
        else:
            my_Attribute[i] = "A135"
    dataset["Attribute13q"] = my_Attribute

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute16[i] == 1:
            my_Attribute[i] = "A161"
        elif dataset.Attribute16[i] == 2:
            my_Attribute[i] = "A162"
        elif dataset.Attribute16[i] == 3:
            my_Attribute[i] = "A163"
        elif dataset.Attribute16[i] == 4:
            my_Attribute[i] = "A164"
    dataset["Attribute16q"] = my_Attribute

    for i in range(0, len(dataset.Id)):
        if dataset.Attribute18[i] == 1:
            my_Attribute[i] = "A181"
        elif dataset.Attribute18[i] == 2:
            my_Attribute[i] = "A182"
    dataset["Attribute18q"] = my_Attribute