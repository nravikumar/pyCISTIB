import numpy as np
import csv
from sklearn.metrics import classification_report
from Utils import Generate_plots


y_true = np.ones((20,1))
y_true = np.concatenate((y_true, np.ones((10,1))*2))
y_true = np.concatenate((y_true, np.zeros((20,1))))


print(y_true.shape)

y_pred = np.ones((20,1))
y_pred = np.concatenate((y_pred, np.ones((10,1))*2))
y_pred = np.concatenate((y_pred, np.zeros((20,1))))

print(y_pred.shape)

labels = [0,1,2]
fig = Generate_plots.PlotLosses.generate_confusion_matrix(y_true,y_pred,labels)

cf_report = classification_report(np.asarray(y_true), np.asarray(y_pred),labels=labels,
                                  target_names=['Full coverage', 'Missing Basal Slice','Missing Apical Slice'],
                                  output_dict=True)

print('Summary of label-wise classification report:\n', cf_report)

# Write classification report dict to csv file
w = csv.writer(open("classification_results.csv", "w"))
for key, val in cf_report.items():
    w.writerow([key,val])