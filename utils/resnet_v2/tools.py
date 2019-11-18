# @Time : 2019/10/11 下午9:05 
# @Author : Xiaoyu Li
# @File : tools.py 
# @Orgnization: Dr.Cubic Lab

import matplotlib.pyplot as plt
import itertools
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

###########################################
## Function to plot confusion matrices  ##
#########################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = np.around(cm, decimals=3)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./result/confusion.eps', format='eps', dpi=1000)

###################################################################
### Callback method for reducing learning rate during training  ###
###################################################################
class AdvancedLearnignRateScheduler(Callback):

    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto', decayRatio=0.1, warmup_batches=-1, init_lr=0.001):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio
        self.warmup_batches = warmup_batches
        self.batch_count = 0
        self.init_lr = init_lr

        if mode not in ['auto', 'min', 'max']:
            # warnings.warn('Mode %s is unknown, '
            #               'fallback to auto mode.'
            #               % (self.mode), RuntimeWarning)
            print('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            # warnings.warn('AdvancedLearnignRateScheduler'
            #               ' requires %s available!' %
            #               (self.monitor), RuntimeWarning)
            print('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
            self.wait += 1

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            # if self.verbose > 0:
            #     print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
            #           'rate to %s.' % (self.batch_count + 1, lr))

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1