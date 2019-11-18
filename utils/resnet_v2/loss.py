# @Time : 2019/10/11 下午6:14
# @Author : Xiaoyu Li
# @File : loss.py
# @Orgnization: Dr.Cubic Lab

import tensorflow as tf
from tensorflow.keras import backend as K

def my_loss(name='unknown'):
    ### YOLO-v1 Loss
    def custom_loss(y_true, y_pred):
        bbox_num = 1
        coord_scale = 5
        no_object_scale = 0.5
        object_scale = 1
        warmup_batches = 0
        GRID_W = 1136
        debug = True
        warmup_stop = False

        mask_shape = tf.shape(y_true)[:1]
        cell_x = tf.cast(tf.reshape(tf.range(GRID_W), (1, GRID_W, 1, 1)), dtype='float')
        cell_grid = tf.tile(cell_x, [16, 1, bbox_num, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        # seen = tf.compat.v2.Variable(0.)
        # total_recall = tf.compat.v2.Variable(0.)

        """
           Adjust prediction
           """
        ### adjust x and y
        pred_box_x = tf.sigmoid(y_pred[..., 0:1]) / bbox_num + cell_grid

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 1:2])

        """
          Adjust ground truth
          """
        ### adjust x and y
        true_box_x = y_true[..., 0:1]  # relative position to the containing cell

        # distance_x = tf.abs(true_box_x - pred_box_x)

        true_box_conf = y_true[..., 1:2]

        """
                Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 1], axis=-1) * coord_scale

        conf_mask = conf_mask + (1 - y_true[..., 1:2]) * no_object_scale
        conf_mask = conf_mask + y_true[..., 1:2] * object_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.cast(coord_mask < coord_scale / 2., dtype='float')
        # seen = tf.assign_add(seen, 1.)
        # seen.assign(1.)
        # if warmup_stop == False:
        #     true_box_x, coord_mask = [true_box_x + (0.5 + cell_grid) * no_boxes_mask, tf.ones_like(coord_mask)]
        #     warmup_stop = True
        # true_box_x, coord_mask = tf.cond(tf.less(seen, warmup_batches + 1),
        #                                            lambda: [true_box_x + (0.5 + cell_grid) * no_boxes_mask,
        #                                                     tf.ones_like(coord_mask)],
        #                                            lambda: [true_box_x,
        #                                                     coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype='float'))
        nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype='float'))

        loss_x = tf.reduce_sum(tf.square(true_box_x - pred_box_x) * coord_mask) / (
                nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (
                nb_conf_box + 1e-6) / 2.
        loss = loss_x + loss_conf
        # loss = tf.cond(tf.less(seen, warmup_batches + 1),
        #                          lambda: loss_x + loss_conf + 10,
        #                          lambda: loss_x + loss_conf)
        if debug:
            nb_true_box = tf.reduce_sum(y_true[..., 1])
            nb_pred_box = tf.reduce_sum(
                tf.cast(true_box_conf >= 0.5, dtype='float') * tf.cast(pred_box_conf > 0.3, dtype='float'))
            current_recall = nb_pred_box / (nb_true_box + 1e-6)
        #     total_recall = total_recall.assign_add(current_recall)
        #     loss = tf.compat.v1.Print(loss, [loss_x], message='\n{}Loss XY \t'.format(name), summarize=1000)
        #     loss = tf.compat.v1.Print(loss, [loss_conf], message='{}Loss Conf \t'.format(name), summarize=1000)
        #     loss = tf.compat.v1.Print(loss, [loss], message='{}Total Loss \t'.format(name), summarize=1000)
        #     loss = tf.compat.v1.Print(loss, [current_recall], message='{}Current Recall \t'.format(name), summarize=1000)
        #     loss = tf.compat.v1.Print(loss, [total_recall / seen], message='{}Average Recall \t'.format(name), summarize=1000)

        return loss

    return custom_loss

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 1 - 2 * ((precision * recall) / (precision + recall + K.epsilon()))
