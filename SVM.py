import platform
import os
import sys
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from numpy import percentile
from pathlib import Path
import time
from scipy.stats import randint as sp_randint
import pickle
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, cohen_kappa_score, make_scorer
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve, SCORERS
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib
from tabulate import tabulate
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import class_weight
from sklearn import svm



NUM_PACKET_THRESHOLD = 400 

#%% Define a TIme Class to computer total execution time
class Timer:
  def __init__(self):
    self.start = time.time()

  def restart(self):
    self.start = time.time()

  def get_time(self):
    end = time.time()
    m, s = divmod(end - self.start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str
#%%Define a correlation plot method
def Correlation_plot(df):
    plt.ioff()
    red_green = ["#ff0000", "#00ff00"]
    sns.set_palette(red_green)
    np.seterr(divide='ignore', invalid='ignore')
    g = sns.pairplot(df,
                     diag_kind = 'kde',
                     hue = 'Altitude',
                     markers = ["o", "D"],
                     size = 1.5,
                     aspect = 1,
                     plot_kws = {"s": 6})
    g.fig.subplots_adjust(right = 0.9)
    plt.show()
#%% Define a method to load the dataset
def LoadData(local_path,model_path):
    global X_train, y_train, X_test, y_test, train, test 
    global feature_columns, response_column, n_features
    subtract_mean = True
    NUM_PACKET_THRESHOLD = 400 
    
    
    hdf5_file = h5py.File(local_path+'dataset.hdf5', "r")
    # Total number of samples
    data_num = hdf5_file["flows"].shape[0]
    flow_rows, flow_cols = NUM_PACKET_THRESHOLD, 24
    X_samples = hdf5_file["flows"][...,0]
    hdf5_file.close()
    
    


    # load labels
    labels = pd.read_hdf(local_path+'dataset.hdf5','labels')
    labels.replace({'label':{'IoT': 1, 'Desktop': 0}}, inplace=True)
    y_samples = labels['label']
    
    
    # read the training mean
    if subtract_mean:
        mm = np.mean(X_samples,axis=0)
        with open(model_path+'mean.pkl', 'wb') as fid:
            pickle.dump(mm, fid)
        mm = mm[np.newaxis, ...]    
        X_samples -= mm
        
        
    train_shape = (X_samples.shape[0], NUM_PACKET_THRESHOLD, 24, 1)
    
    X_samples = np.nan_to_num(X_samples)
    print(np.isnan(X_samples).sum())
    
    
    # extract quantiles from flows 
    buf = np.zeros(24*5, np.float32)
    x_train_n = np.zeros((X_samples.shape[0],120),np.float32)    
    for i in range(0, X_samples.shape[0]):
        for j in range(0, X_samples.shape[2]):
            data = X_samples[i,:,j]
            quartiles = percentile(data, [25, 50, 75])
            data_min, data_max = data.min(), data.max()
            buf[(j*5+0):(j*5+5)] = data_min, quartiles[0],quartiles[1],quartiles[2], data_max
        x_train_n[i,:] = buf 
    
    # Split the data into Train and Test with Train having 80% and test 20% each
    X_train, X_test, y_train, y_test = train_test_split(x_train_n, y_samples, test_size = 0.2, shuffle=True)   
  
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
           
    
    return
#%% Define a method to plot a ROC Curve

def ROC_Curve(rf, auc):
    one_hot_encoder = OneHotEncoder()
    rf_fit = rf.fit(X_train, y_train)
    fit = one_hot_encoder.fit(rf.apply(X_train))
    y_predicted = rf.predict_proba(X_test)[:, 1]
    false_positive, true_positive, threshold = roc_curve(y_test, y_predicted)
    
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive, true_positive, color='darkorange', label='Random Forest')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (area = %0.2f)' % auc)
    plt.legend(loc='best')
    plt.show()
#%% Define a method to print the model performance

def Print_Metrics(saved_rf):
    print('\nModel performance on the test data set:')

    # print('Train Accuracy.......', accuracy_score(y_train, best_model.predict(X_train)))
    # print('Validate Accuracy....', accuracy_score(y_valid, best_model.predict(X_valid)))

    y_predict_test  = best_model.predict(X_test)
    mse             = metrics.mean_squared_error(y_test, y_predict_test)
    logloss_test    = metrics.log_loss(y_test, y_predict_test)
    accuracy_test   = metrics.accuracy_score(y_test, y_predict_test)
    accuracy_test2  = best_model.score(X_test, y_test)
    F1_test         = metrics.f1_score(y_test, y_predict_test)
    precision_test  = precision_score(y_test, y_predict_test, average='binary')
    precision_test2 = metrics.precision_score(y_test, y_predict_test)
    recall_test     = recall_score(y_test, y_predict_test, average='binary')
    auc_test        = metrics.roc_auc_score(y_test, y_predict_test)
    r2_test         = metrics.r2_score(y_test, y_predict_test)
    
    header = ["Metric", "Test"]
    table  = [
               ["logloss",   logloss_test],
               ["accuracy",  accuracy_test],
               ["precision", precision_test],
               ["F1",        F1_test],
               ["r2",        r2_test],
               ["AUC",       auc_test]
             ]

    print(tabulate(table, header, tablefmt="fancy_grid"))
#%% Define a method to plot the predictor importance

def Plot_predictor_importance(best_model, feature_columns):
    feature_importance = best_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    y_pos  = np.arange(sorted_idx.shape[0]) + .5
    fig, ax = plt.subplots()
    ax.barh(y_pos, 
            feature_importance[sorted_idx], 
            align='center', 
            color='green', 
            ecolor='black', 
            height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_columns)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance')
    ax.set_title('Predictor Importance')
    plt.show()

#%% Define a utility function to report best scores

def Report_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


#%% Define a method to print the Confusion Matrix and the performance metrics
def Print_confusion_matrix(cm, auc, heading):
    print('\n', heading)
    print(cm)
    true_negative  = cm[0,0]
    true_positive  = cm[1,1]
    false_negative = cm[1,0]
    false_positive = cm[0,1]
    total = true_negative + true_positive + false_negative + false_positive
    accuracy = (true_positive + true_negative)/total
    precision = (true_positive)/(true_positive + false_positive)
    recall = (true_positive)/(true_positive + false_negative)
    misclassification_rate = (false_positive + false_negative)/total
    F1 = (2*true_positive)/(2*true_positive + false_positive + false_negative)
    print('accuracy.................%7.4f' % accuracy)
    print('precision................%7.4f' % precision)
    print('recall...................%7.4f' % recall)
    print('F1.......................%7.4f' % F1)
    print('auc......................%7.4f' % auc)

#%% Plot the learning curves

def Plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        n_jobs = 1, train_sizes = np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
                                                            X, y,
                                                            cv = cv,
                                                            n_jobs = n_jobs,
                                                            train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return
#%% Define the hyperparameters for a random search

def Random_Search(model_path):
    global best_model, saved_moldel

    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}

    clf = svm.SVC(class_weight = 'balanced')
    n_iter_search = 5                                                  
    estimator = RandomizedSearchCV(clf,
                                   param_distributions = param_grid,
                                   n_iter = n_iter_search,
                                   scoring = 'roc_auc',#'neg_log_loss',
                                   verbose = 0,                         
                                   n_jobs = -1)                          
    
    
    fit = estimator.fit(X_train, y_train)

    # Cross validation with 20 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv_ = ShuffleSplit(n_splits = 20, test_size = 0.20, random_state = 0)
     
    Report_scores(estimator.cv_results_, n_top = 3)
    
    best_model = estimator.best_estimator_
    
    y_predicted = best_model.predict(X_train)

    c_report = classification_report(y_train, y_predicted)
    print('\nClassification report:\n', c_report)

    y_predicted_train = best_model.predict(X_train)
    cm = confusion_matrix(y_train, y_predicted_train)
    auc = roc_auc_score(y_train, y_predicted_train)
    Print_confusion_matrix(cm, auc, 'Confusion matrics of the training dataset')

    y_predicted = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_predicted)
    auc = roc_auc_score(y_test, y_predicted)

    ntotal = len(y_test)
    correct = y_test == y_predicted
    numCorrect = sum(correct)
    percent = round( (100.0*numCorrect)/ntotal, 6)
    print("\nCorrect classifications on test data: {0:d}/{1:d} {2:8.3f}%".format(numCorrect, ntotal, percent))
    prediction_score = 100.0*best_model.score(X_test, y_test)
    print('Random Forest Prediction Score on test data: %8.3f' % prediction_score)

    # saving the best model
    joblib.dump(best_model, model_path+"GNB-dist.pkl")

    saved_model = joblib.load(model_path+"GNB-dist.pkl")
    y_predicted_test = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_predicted_test)
    auc = roc_auc_score(y_test, y_predicted_test)
    Print_confusion_matrix(cm, auc, 'Confusion matrics of the test dataset')
    #ROC_Curve(best_model, auc)
    Print_Metrics(saved_model)

    
    return
#%% Run the random search

def modelTraining(local_path,model_path):
    print('Operating system version....', platform.platform())
    print("Python version is........... %s.%s.%s" % sys.version_info[:3])
    print('scikit-learn version is.....', sklearn.__version__)
    print('pandas version is...........', pd.__version__)
    print('numpy version is............', np.__version__)
    print('matplotlib version is.......', matplotlib.__version__)

    my_timer = Timer()
    LoadData(local_path,model_path)
    print('Data loaded!!!!')
    Random_Search(model_path)
    elapsed = my_timer.get_time()
    print("\nTotal compute time was: %s" % elapsed)



