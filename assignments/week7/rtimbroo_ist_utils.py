# -*- coding: utf-8 -*-
'''
Basic Logger
'''
def getSHLogger(name='stream_handler',level=20):
    """
    name: Get a Logger
    
    Args:
        param1 (int): Log level
    
    Returns:
        Returns a logger object
    
    """
    import logging
    #print(level)
    loglevel = None
    if level == 10: # DEBUG
        loglevel = logging.DEBUG;
    elif level == 20: # INFO
        loglevel = logging.INFO;
    elif level == 30: # WARNING
        loglevel = logging.WARNING;
    elif level == 40: # ERROR
        loglevel = logging.ERROR;
    elif level == 50: # CRITICAL
        loglevel = logging.CRITICAL;
    else:
        loglevel = logging.DEBUG;
    
    
    isSimpleOutput = True
    l = logging.getLogger(name)
    
    if not l.hasHandlers():
        f = None
        l.setLevel(loglevel)
        h = logging.StreamHandler()
        if isSimpleOutput:
            f = logging.Formatter('%(message)s')
        else:
            f = logging.Formatter('Date Time: %(asctime)s | Level: %(levelname)s | Message: %(message)s')
        
        h.setFormatter(f)
        l.addHandler(h)
        l.setLevel(loglevel)
        l.handler_set = True
        
    return l
'''
Find NaN values in dataframe
'''
def getNaNCount(df):
    
    totNaNCnt = df.isnull().sum().sum()
    nanRowsCnt = len(df[df.isnull().T.any().T])
    
    #print("Total NaN Cnt {0}".format(totNaNCnt))
    #print("Total NaN Rows Cnt {0}".format(nanRowsCnt))
    
    return totNaNCnt, nanRowsCnt
    
'''

'''
def findColumnsNaN(df,rowIndex=True):
    naCols = []
    for col in list(df.columns):
        #print(coachesDf[col].isnull().sum().sum())
        if df[col].isnull().sum().sum() > 0:
            print("Column: {0} has: {1} NaN values".format(col,df[col].isnull().sum().sum()))
            if rowIndex: print("{0}: {1}\n".format(col,getNaNIndexes(df,col)))
   
'''

'''  
def getColumnsNaNCnts(df,rowIndex=True):
    naCols = []
    for col in list(df.columns):
        #print(coachesDf[col].isnull().sum().sum())
        if df[col].isnull().sum().sum() > 0:
            naCols.append((col,df[col].isnull().sum().sum()))
    
    #print(len(naCols))
    if not len(naCols) == 0:
        return(naCols)
    else:
        return(0)
           
'''

'''
def getNaNIndexes(df,att):
    import numpy as np
    n = np.where(df[att].isnull()==True)
    return list(n[0])


'''
# find missing data
'''
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

'''
convert NaN values to means of the column values
'''
def nan2Mean(df):
    for x in list(df.columns.values):
        #print("___________________"+x)
        #print(df[x].isna().sum())
        df[x] = df[x].fillna(df[x].mean())
       #print("Mean-"+str(df[x].mean()))
    return df


'''
perform label encoding
'''
def labelEncoding(train_df, test_df):
    from sklearn.preprocessing import LabelEncoder
    # Label Encoding
    for f in train_df.columns:
        if  train_df[f].dtype=='object': 
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))  
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    return(train_df,test_df)

'''

'''
#black box LGBM 
def LGB_bayesian(num_leaves, bagging_fraction,feature_fraction,min_child_weight, min_data_in_leaf,max_depth,reg_alpha,reg_lambda ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    

    param = {
              'num_leaves': num_leaves, 
              'min_data_in_leaf': min_data_in_leaf,
              'min_child_weight': min_child_weight,
              'bagging_fraction' : bagging_fraction,
              'feature_fraction' : feature_fraction,
              #'learning_rate' : learning_rate,
              'max_depth': max_depth,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'objective': 'binary',
              'save_binary': True,
              'seed': 1337,
              'feature_fraction_seed': 1337,
              'bagging_seed': 1337,
              'drop_seed': 1337,
              'data_random_seed': 1337,
              'boosting_type': 'gbdt',
              'verbose': 1,
              'is_unbalance': False,
              'boost_from_average': True,
              'metric':'auc'}    
    
    oof = np.zeros(len(train_df))
    trn_data= lgb.Dataset(train_df.iloc[bayesian_tr_idx][features].values, label=train_df.iloc[bayesian_tr_idx][target].values)
    val_data= lgb.Dataset(train_df.iloc[bayesian_val_idx][features].values, label=train_df.iloc[bayesian_val_idx][target].values)

    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)
    
    oof[bayesian_val_idx]  = clf.predict(train_df.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  
    
    score = roc_auc_score(train_df.iloc[bayesian_val_idx][target].values, oof[bayesian_val_idx])

    return score

'''
# Confusion matrix
'''
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues):
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

