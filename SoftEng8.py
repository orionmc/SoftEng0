import csv
import pandas as pd
import sweetviz as sv
#import pycaret
from pycaret.classification import *


# # Ignore all warnings
# import warnings
# warnings.filterwarnings("ignore")


# Loading the dataset
# heartds0 = pd.read_csv('https://raw.githubusercontent.com/orionmc/SoftEng0/main/heart.ds.csv')
heartds0 = pd.read_csv(r'C:\Users\user0\Documents\BPP\Software Engineering\Git\0\SoftEng0\SoftEng0\heart.ds.csv')


report = sv.analyze(heartds0, target_feat='output')
report.show_notebook()

s = setup(heartds0,                     # data set
          target = 'output',            # the goal for prediction
          ignore_features = ['sex'],    # we may ignore certaion futures if deement irrelevant
          session_id = 123)


