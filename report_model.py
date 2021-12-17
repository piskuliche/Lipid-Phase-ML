#!/usr/bin/env python
import numpy
import pickle
import sys,mllpa
import subprocess

modelname=str(sys.argv[1])

final_model = pickle.load(open(modelname,'rb'))
for key in final_model['scores']['final_score']:
    print("%s : %s" % (key, final_model['scores']['final_score'][key]))


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
tree.export_graphviz(final_model['ClassificationTree'],out_file="viz.out",feature_names=['SVM-Coords','KNN-Coords','SVM-Distances','NB'],class_names=['gel','fluid'],rounded=True,rotate=True,proportion=True,filled=True)

subprocess.run("dot -Tpng viz.out -o viz.png",shell=True)
