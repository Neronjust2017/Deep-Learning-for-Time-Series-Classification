from builtins import print
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist[metric])
    plt.plot(hist['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def save_logs(output_directory,  y_true,duration,lr=True,y_true_val=None,y_pred_val=None):
    hist_df = np.load('history.csv')

    # hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history1.csv', index=False)

    # df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val,y_pred_val)
    # df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0],
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # 最好 val_loss 模型
    index_best_model_val = hist_df['val_loss'].idxmin()
    row_best_model_val = hist_df.loc[index_best_model_val]

    df_best_model_val  = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model_val ['best_model_train_loss'] = row_best_model_val ['loss']
    df_best_model_val ['best_model_val_loss'] = row_best_model_val ['val_loss']
    df_best_model_val ['best_model_train_acc'] = row_best_model_val ['accuracy']
    df_best_model_val ['best_model_val_acc'] = row_best_model_val ['val_accuracy']
    if lr == True:
        df_best_model_val ['best_model_learning_rate'] = row_best_model_val ['lr']
    df_best_model_val ['best_model_nb_epoch'] = index_best_model_val

    df_best_model_val .to_csv(output_directory + 'df_best_model_val.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist_df,output_directory+'epochs_loss.png')

    return

root_dir = 'data'
output_directory = root_dir + '/results/fcn/TSC_itr_8/AAecg'

save_logs(output_directory)