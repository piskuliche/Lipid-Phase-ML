#!/usr/bin/env python
import numpy
import pickle
import sys,mllpa

modelname=str(sys.argv[1])

final_model = pickle.load(open(modelname,'rb'))
for key in final_model['scores']['final_score']:
    print("%s : %s" % (key, final_model['scores']['final_score'][key]))
