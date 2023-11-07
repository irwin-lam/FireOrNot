import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from src.visualizations import plot_cm

class metric_note():

    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid
        self.printout = pd.DataFrame({'Model': pd.Series(dtype='str'),
                                      'Size': pd.Series(dtype='int'),
                                      'train log_loss': pd.Series(dtype='float64'),
                                      'train accuracy': pd.Series(dtype='float64'),
                                      'train TP': pd.Series(dtype='int32'),                        
                                      'train TN': pd.Series(dtype='int32'),
                                      'train FP': pd.Series(dtype='int32'),
                                      'train FN': pd.Series(dtype='int32'),
                                      'test log_loss': pd.Series(dtype='float64'),
                                      'test accuracy': pd.Series(dtype='float64'),
                                      'test TP': pd.Series(dtype='int32'),                            
                                      'test TN': pd.Series(dtype='int32'),
                                      'test FP': pd.Series(dtype='int32'),
                                      'test FN': pd.Series(dtype='int32'),
                                      'val log_loss': pd.Series(dtype='float64'),
                                      'val accuracy': pd.Series(dtype='float64'),
                                      'val TP': pd.Series(dtype='int32'),                                      
                                      'val TN': pd.Series(dtype='int32'),
                                      'val FP': pd.Series(dtype='int32'),
                                      'val FN': pd.Series(dtype='int32'),})
        
    def evaluate(self, model, name, size):
        train_results = model.evaluate(self.train)
        test_results = model.evaluate(self.test)
        valid_results = model.evaluate(self.valid)

        add = [name, size] + train_results + test_results + valid_results
        self.printout.loc[len(self.printout.index)] = add
        self.printout

        plot_cm(name, test_results, size)