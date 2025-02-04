import pandas as pd
import time
import csv
from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np
from nltk.stem import PorterStemmer
import re
import xgboost as xgb

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
                                         'renov','air', 'walkin', 'duplex', 'common', 'elev', 'unknown', 'display_address']]

train_df_vector['manager_id'] = train_df_vector['manager_id'].astype('category')
train_df_vector['street_address'] = train_df_vector['street_address'].astype('category')
train_df_vector['display_address'] = train_df_vector['display_address'].astype('category')

categorical_columns = train_df_vector.select_dtypes(['category']).columns
train_df_vector[categorical_columns] = train_df_vector[categorical_columns].apply(lambda x: x.cat.codes)

train_df_target = train_df_total.loc[:, 'interest_level']
print("Done with training data")

test_df = pd.read_json('test.json')
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
#
# param = dict()
# param['objective'] = 'multi:softprob'
# param['max_depth'] = 6
# param['silent'] = False
# param['num_class'] = 3
# param['eval_metric'] = "mlogloss"
# param['min_child_weight'] = 1
# param['subsample'] = 0.7
# param['colsample_bylevel'] = 0.7
# param['seed'] = 350
# param['n_estimators'] = 2000

# plst = list(param.items())

reg = xgb.XGBClassifier(objective='multi:softprob', max_depth=6, silent=False, min_child_weight=1, subsample=0.7,
                        colsample_bylevel=0.7, seed=312, n_estimators=2000)
reg.fit(train_df_vector, train_df_target)
predict = reg.predict_proba(test_df_vector)

print("Time for the XGBoost Classifier to train and predict on the testing data is := %.2f" % (time.time() -
                                                                                                     start_time))

csv_file = open("submissions_new_xgboost.csv", 'w')
wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
wr.writerow(['listing_id', 'high', 'medium', 'low'])

for index in range(0, len(test_df_ids)):
    wr.writerow([test_df_ids[index], predict[index][0], predict[index][2], predict[index][1]])
    index += 1

print("Done with predicting Interest Levels for the test data")
csv_file.close()
