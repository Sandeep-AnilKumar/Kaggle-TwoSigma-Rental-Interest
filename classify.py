import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
from sklearn import svm
import csv
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

start_time = time.time()
train_df_total = pd.read_json('train.json')

# train_high = train_df.loc[train_df['interest_level'] == 'high'].sample(frac=1)
# train_medium = train_df.loc[train_df['interest_level'] == 'medium'].sample(frac=0.34)
# train_low = train_df.loc[train_df['interest_level'] == 'low'].sample(frac=0.115)
#
# frames = [train_high, train_medium, train_low]
#
# train_df_total = pd.concat(frames, ignore_index=True)

print "Number of high interest houses := " + str(len(train_df_total.loc[train_df_total['interest_level'] == 'high']))
print "Number of medium interest houses := " + str(len(train_df_total.loc[train_df_total['interest_level'] == 'medium']))
print "Number of low interest houses := " + str(len(train_df_total.loc[train_df_total['interest_level'] == 'low']))

train_df_total['num_photos'] = train_df_total['photos'].apply(len)
train_df_total['num_features'] = train_df_total['features'].apply(len)
train_df_total['num_description'] = train_df_total['description'].apply(lambda x: len(x.split(" ")))

train_df_vector = train_df_total.loc[:, ['bathrooms', 'bedrooms', 'num_photos', 'num_features', 'num_description',
                                         'building_id', 'display_address', 'latitude', 'longitude', 'manager_id',
                                         'price', 'street_address']]

train_df_vector['building_id'] = train_df_vector['building_id'].astype('category')
train_df_vector['display_address'] = train_df_vector['display_address'].astype('category')
train_df_vector['manager_id'] = train_df_vector['manager_id'].astype('category')
train_df_vector['street_address'] = train_df_vector['street_address'].astype('category')

categorical_columns = train_df_vector.select_dtypes(['category']).columns
train_df_vector[categorical_columns] = train_df_vector[categorical_columns].apply(lambda x: x.cat.codes)

train_df_target = train_df_total.loc[:, 'interest_level']

test_df = pd.read_json('test.json')

test_df['num_photos'] = test_df['photos'].apply(len)
test_df['num_features'] = test_df['features'].apply(len)
test_df['num_description'] = test_df['description'].apply(lambda x: len(x.split(" ")))

test_df_vector = test_df.loc[:, ['bathrooms', 'bedrooms', 'num_photos', 'num_features', 'num_description',
                                 'building_id', 'display_address', 'latitude', 'longitude', 'manager_id', 'price',
                                 'street_address']]

test_df_vector['building_id'] = test_df_vector['building_id'].astype('category')
test_df_vector['display_address'] = test_df_vector['display_address'].astype('category')
test_df_vector['manager_id'] = test_df_vector['manager_id'].astype('category')
test_df_vector['street_address'] = test_df_vector['street_address'].astype('category')

categorical_columns = test_df_vector.select_dtypes(['category']).columns
test_df_vector[categorical_columns] = test_df_vector[categorical_columns].apply(lambda x: x.cat.codes)

test_df_ids = list(test_df['listing_id'])

#Gradient Boost Classifier
#gbc_clf = gbc(loss='deviance')

# SGDC classifier
# sgd_clf = linear_model.SGDClassifier(loss='log')
# #
# # # KNN classifier
# # knn_clf = KNeighborsClassifier()
# #
# #logistic regression
# lr_clf = linear_model.LogisticRegression()
#
# # Naive Bayes
# nb_clf = BernoulliNB()
#

# ab_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5,
#                                   algorithm="SAMME.R")

# Ensemble classifier, taking best 3 classifiers out of the above 5.
# voting_clf = VotingClassifier(estimators=[('sgd', sgd_clf), ('lr', lr_clf), ('ab_clf', ab_clf)],
#                               weights=[0.5, 0.5, 1], voting='soft')

rfc_clf = rfc(n_estimators=1000)

rfc_clf.fit(train_df_vector, train_df_target)
voting_prediction = rfc_clf.predict_proba(test_df_vector)

print("Time for the Voting Classifier to train and predict on the testing data subset is := %.2f" % (time.time() -
                                                                                                     start_time))

csv_file = open("submissions.csv", 'w')
wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
wr.writerow(['listing_id', 'high', 'medium', 'low'])

for index in range(0, len(test_df_ids)):
    wr.writerow([test_df_ids[index], voting_prediction[index][0], voting_prediction[index][2], voting_prediction[index][1]])
    index += 1

print("Done with predicting Interest Levels for the test data")
csv_file.close()
