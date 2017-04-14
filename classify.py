import pandas as pd
import time
import csv
from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

np.seterr(divide='ignore', invalid='ignore')
stemmer = PorterStemmer()

features = ['garag', 'cinema', 'hardwood', 'air condit', 'modern', 'share', 'dog', 'landlord', 'unassign', '24 hour',
            'fireplac', 'onsit', 'cat', 'storag', 'spaciou', 'huge', 'privat', 'fee', 'yoga', 'fit', 'pet', 'open',
            'dishwash', 'walk', 'attach', 'state', 'includ', 'inunit', 'new', 'washer', 'gym', 'deposit', 'AC', 'queen',
            'private terrac', 'twentyfour', 'attend', 'underpr', 'brand', 'upon', 'valet', 'free', 'courtyard', 'dryer',
            'laundri', 'ok', 'granit', 'overs', 'pool', 'avail', 'convert', 'onli', 'central', 'gift', 'wifi', 'renov',
            'air', 'walkin', 'duplex', 'common', 'elev', 'unknown']

start_time = time.time()
train_df_total = pd.read_json('train.json')
test_df = pd.read_json('test.json')
for feature in features:
    train_df_total[feature] = 0

def add_features(fe):
    found = False
    f_list = train_df_total.loc[fe, "features"]
    if f_list:
        for f in f_list:
            #print f
            #f = f.encode('utf-8').strip()
            for f_word in f.split(" "):
                #print f_word
                f_word = f_word.lower().strip()
                f_word = f_word.replace("\"", "")
                f_word = f_word.replace("'", "")
                f_word = re.sub(r'(#\*!,\.\-)*', "", f_word)
                for fea in features:
                    if stemmer.stem(f_word) in fea:
                        #print "found " + str(f_word) + " in " + str(fea)
                        train_df_total.loc[fe, fea] = 1
                        found = True
                        break
                if found:
                    break
    return

def add_test_features(fe):
    found = False
    f_list = test_df.loc[fe, "features"]
    if f_list:
        for f in f_list:
            #print f
            #f = f.encode('utf-8').strip()
            for f_word in f.split(" "):
                #print f_word
                f_word = f_word.lower().strip()
                f_word = f_word.replace("\"", "")
                f_word = f_word.replace("'", "")
                f_word = re.sub(r'(#\*!,\.\-)*', "", f_word)
                for fea in features:
                    if stemmer.stem(f_word) in fea:
                        #print "found " + str(f_word) + " in " + str(fea)
                        test_df.loc[fe, fea] = 1
                        found = True
                        break
                if found:
                    break
    return

fea_list = train_df_total.index.values.tolist()

#print train_df_total.loc[10000, "features"]
for fea_l in fea_list:
    #print fea_l
    add_features(fea_l)

#print train_df_total.loc[10000, ["elev", "dog", "cat", "fit", "renov"]]

train_df_total['num_photos'] = train_df_total['photos'].apply(len)
train_df_total['num_features'] = train_df_total['features'].apply(len)
train_df_total['num_description'] = train_df_total['description'].apply(lambda x: len(x.split(" ")))

train_df_vector = train_df_total.loc[:, ['bathrooms', 'bedrooms', 'num_photos', 'num_features', 'num_description',
                                         'latitude', 'longitude', 'manager_id','price', 'street_address', 'garag',
                                         'cinema', 'hardwood', 'air condit', 'modern', 'share', 'dog', 'landlord',
                                         'unassign', '24 hour','fireplac', 'onsit', 'cat', 'storag', 'spaciou',
                                         'huge', 'privat', 'fee', 'yoga', 'fit', 'pet', 'open','dishwash', 'walk',
                                         'attach', 'state', 'includ', 'inunit', 'new', 'washer', 'gym', 'deposit',
                                         'AC', 'queen','private terrac', 'twentyfour', 'attend', 'underpr', 'brand',
                                         'upon', 'valet', 'free', 'courtyard', 'dryer','laundri', 'ok', 'granit',
                                         'overs', 'pool', 'avail', 'convert', 'onli', 'central', 'gift', 'wifi',
                                         'renov','air', 'walkin', 'duplex', 'common', 'elev', 'unknown', 'display_address',
                                         'interest_level']]

train_df_vector['manager_id'] = train_df_vector['manager_id'].astype('category')
train_df_vector['street_address'] = train_df_vector['street_address'].astype('category')
train_df_vector['display_address'] = train_df_vector['display_address'].astype('category')
train_df_vector['interest_level'] = train_df_vector['interest_level'].astype('category')

categorical_columns = train_df_vector.select_dtypes(['category']).columns
train_df_vector[categorical_columns] = train_df_vector[categorical_columns].apply(lambda x: x.cat.codes)

train_df_target = train_df_total.loc[:, 'interest_level']
print("Done with training data")

Y = train_df_vector.pop('interest_level')
X = train_df_vector.as_matrix()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#train = train_df_vector.sample(frac=0.8,random_state=200)
#test = train_df_vector.drop(train.index)

for feature in features:
    test_df[feature] = 0

fea_list = test_df.index.values.tolist()

#print train_df_total.loc[10000, "features"]
for fea_l in fea_list:
    #print fea_l
    add_test_features(fea_l)

#print train_df_total.loc[10000, ["elev", "dog", "cat", "fit", "renov"]]

test_df['num_photos'] = test_df['photos'].apply(len)
test_df['num_features'] = test_df['features'].apply(len)
test_df['num_description'] = test_df['description'].apply(lambda x: len(x.split(" ")))

test_df_vector = test_df.loc[:, ['bathrooms', 'bedrooms', 'num_photos', 'num_features', 'num_description',
                                         'latitude', 'longitude', 'manager_id','price', 'street_address', 'garag',
                                         'cinema', 'hardwood', 'air condit', 'modern', 'share', 'dog', 'landlord',
                                         'unassign', '24 hour','fireplac', 'onsit', 'cat', 'storag', 'spaciou',
                                         'huge', 'privat', 'fee', 'yoga', 'fit', 'pet', 'open','dishwash', 'walk',
                                         'attach', 'state', 'includ', 'inunit', 'new', 'washer', 'gym', 'deposit',
                                         'AC', 'queen','private terrac', 'twentyfour', 'attend', 'underpr', 'brand',
                                         'upon', 'valet', 'free', 'courtyard', 'dryer','laundri', 'ok', 'granit',
                                         'overs', 'pool', 'avail', 'convert', 'onli', 'central', 'gift', 'wifi',
                                         'renov','air', 'walkin', 'duplex', 'common', 'elev', 'unknown', 'display_address']]

test_df_vector['manager_id'] = test_df_vector['manager_id'].astype('category')
test_df_vector['street_address'] = test_df_vector['street_address'].astype('category')
test_df_vector['display_address'] = test_df_vector['display_address'].astype('category')

categorical_columns = test_df_vector.select_dtypes(['category']).columns
test_df_vector[categorical_columns] = test_df_vector[categorical_columns].apply(lambda x: x.cat.codes)

test_df_ids = list(test_df['listing_id'])
print("Done with testing data")

estimator = rfc()
param_grid = {"n_estimators": [10, 20, 50, 80, 100, 200, 250, 300, 500, 600, 900, 1000, 2500, 5000, 7500, 10000],
              "criterion": ["gini", "entropy"], "max_features": [3, 5],"max_depth": [10, 20],
              "min_samples_split": [2, 4], "bootstrap": [True, False]}

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid)
classifier.fit(x_train, y_train)

print "best n_estimators := " + str(classifier.best_params_)
#
rfc_clf = rfc(n_estimators=classifier.best_estimator_.n_estimators)
rfc_clf.fit(Y, train_df_target)

voting_prediction = rfc_clf.predict_proba(test_df_vector)

print("Time for the Random Forest Classifier to train and predict on the testing data is := %.2f" % (time.time() -
                                                                                                     start_time))

csv_file = open("submissions_new.csv", 'w')
wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
wr.writerow(['listing_id', 'high', 'medium', 'low'])

for index in range(0, len(test_df_ids)):
    wr.writerow([test_df_ids[index], voting_prediction[index][0], voting_prediction[index][2], voting_prediction[index][1]])
    index += 1

print("Done with predicting Interest Levels for the test data")
csv_file.close()
