import pandas as pd

class metric_note():

    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid
        self.printout = pd.DataFrame({'Model': pd.Series(dtype='str'),
                                      'Size': pd.Series(dtype='int'),
                                      'train log_loss': pd.Series(dtype='float64'),
                                      'train accuracy': pd.Series(dtype='float64'),                        
                                      'train precision': pd.Series(dtype='float64'),
                                      'train recall': pd.Series(dtype='float64'),
                                      'train auc': pd.Series(dtype='float64'),
                                      'test log_loss': pd.Series(dtype='float64'),
                                      'test accuracy': pd.Series(dtype='float64'),                            
                                      'test precision': pd.Series(dtype='float64'),
                                      'test recall': pd.Series(dtype='float64'),
                                      'test auc': pd.Series(dtype='float64'),
                                      'val log_loss': pd.Series(dtype='float64'),
                                      'val accuracy': pd.Series(dtype='float64'),                                      
                                      'val precision': pd.Series(dtype='float64'),
                                      'val recall': pd.Series(dtype='float64'),
                                      'val auc': pd.Series(dtype='float64'),})
        
    def evaluate(self, model, name, size):
        train_results = model.evaluate(self.train)
        test_results = model.evaluate(self.test)
        valid_results = model.evaluate(self.valid)

        add = [name, size] + train_results + test_results + valid_results
        self.printout.loc[len(self.printout.index)] = add
        self.printout