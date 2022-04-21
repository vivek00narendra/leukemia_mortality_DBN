# DBN framework created by albertbup
# https://github.com/albertbup/deep-belief-network 

import numpy as np

np.random.seed(1337)  # for reproducibility
import sklearn
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dbn import SupervisedDBNClassification


import csv

ds9 = csv.reader(open("ds9_class_cvs.csv"), delimiter=",")
allsnp = csv.reader(open("allsnp_test_replace.csv"), delimiter=",")

# Reading ds9 data
ds9_header = []
ds9_header = next(ds9)

ds9_rows = []
for row in ds9:
    ds9_rows.append(row)

# Reading allsnp data
allsnp_header = []
allsnp_header = next(allsnp)

allsnp_rows = []
for row in allsnp:
    allsnp_rows.append(row)

# # Processed allsnp excluding repeat
# processed_allsnp = []
# for i in range(len(allsnp_rows)):
#     for j in range(len(ds9_rows)):
#         if allsnp_rows[i][0] == ds9_rows[j][0]:
#             temp = allsnp_rows[i]
#             temp.pop(0)
#             for k in range(len(temp)):
#               # Replace missing values with 0.5
#               if(temp[k] == 'NA'):
#                 temp[k] = 0.5
#                 break
#               num = float(temp[k].replace(",","."))
#               temp[k] = float(num)

#             processed_allsnp.append([ds9_rows[j][0], ds9_rows[j][4], temp])

# Processed allsnp including repeat
processed_allsnp = []
for i in range(len(allsnp_rows)):
    for j in range(len(ds9_rows)):
        if str(allsnp_rows[i][0]).replace('.R', '') == str(ds9_rows[j][0]):
            temp = allsnp_rows[i]
            temp.pop(0)
            for k in range(len(temp)):
              # Replace missing values with 0.5
              if(temp[k] == 'NA'):
                temp[k] = 0.5
                break
              num = float(temp[k].replace(",","."))
              temp[k] = float(num)

            processed_allsnp.append([ds9_rows[j][0], ds9_rows[j][4], temp])

# Changing 'y' and 'n' values to 1 and 0 for mortality
for i in range(len(processed_allsnp)):
  if processed_allsnp[i][1] == 'N':
    processed_allsnp[i][1] = 0 
  elif processed_allsnp[i][1] == 'Y':
    processed_allsnp[i][1] = 1

# Normalizing Values
for i in processed_allsnp:
    temp = np.array(i[2])
    normalized_temp = preprocessing.normalize([temp])
    normalized_temp = normalized_temp[0]
    normalized_temp = normalized_temp.tolist()
    i[2] = normalized_temp

# Training data and mortality values
training_data = []
mortality_data = []
for i in range(len(processed_allsnp)):
  training_data.append(processed_allsnp[i][2])

for i in range(len(processed_allsnp)):
  mortality_data.append(processed_allsnp[i][1])

training_data = np.array(training_data)
mortality_data = np.array(mortality_data)
print(np.shape(training_data))
print(np.shape(mortality_data))

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(training_data, mortality_data, test_size=0.2, random_state=0)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(Y_train))
print(np.shape(Y_test))


# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 164],
                                         learning_rate_rbm=0.01,
                                         learning_rate=0.05,
                                         n_epochs_rbm=20,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)


# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
