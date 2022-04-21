# DBN framework created by albertbup
# https://github.com/albertbup/deep-belief-network 

import numpy as np
import csv
import re

np.random.seed(1337)  # for reproducibility
import sklearn
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dbn import SupervisedDBNClassification

ds9 = csv.reader(open("ds9_class_cvs.csv"), delimiter=",")
patient_id = csv.reader(open("patientids.csv"), delimiter=",")
allaffy = csv.reader(open("allaffy_test.csv"), delimiter=",")

# Reading ds9 data
ds9_header = []
ds9_header = next(ds9)

ds9_rows = []
for row in ds9:
    ds9_rows.append(row)

# Reading allaffy data:
allaffy_header = []
allaffy_header = next(allaffy)

allaffy_rows = []
for row in allaffy:
    allaffy_rows.append(row)

# Reading patient ID data:
patient_id_rows = []
for row in patient_id:
    patient_id_rows.append(row)

allaffy_header.insert(0, 'patient_ID')
for i in range(len(patient_id_rows)):
    allaffy_rows[i].insert(0, patient_id_rows[i])


processed_allaffy = []

repeat_indices = [20, 45, 50, 55, 61, 70, 73, 76]
repeat_ids = [123.1, 293.1, 3.1, 308.1, 313.1, 326.1, 33.1, 338.1]

# Without Repeat patients
# Delete repeat values 
for i in repeat_indices:
    del allaffy_rows[i]

for i in range(len(allaffy_rows)):
    allaffy_rows[i][0] = allaffy_rows[i][0][0]
    temp = allaffy_rows[i]
    temp.pop(0)
    processed_allaffy.append([ds9_rows[i][0], ds9_rows[i][4], temp])

# Changing 'y' and 'n' values to 1 and 0 for mortality
for i in range(len(processed_allaffy)):
  if processed_allaffy[i][1] == 'N':
    processed_allaffy[i][1] = 0 
  elif processed_allaffy[i][1] == 'Y':
    processed_allaffy[i][1] = 1

# Normalizing values:
for i in processed_allaffy:
    temp = np.array(i[2])
    normalized_temp = preprocessing.normalize([temp])
    normalized_temp = normalized_temp[0]
    normalized_temp = normalized_temp.tolist()
    i[2] = normalized_temp

# Training data and mortality values
training_data = []
mortality_data = []
for i in range(len(processed_allaffy)):
  training_data.append(processed_allaffy[i][2])

for i in range(len(processed_allaffy)):
  mortality_data.append(processed_allaffy[i][1])

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
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 149],
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
