'''
Special Sequential data generator class which enables using multithreading amd multiprocessing in fit_generator method of keras model
If sequential generator is not used during training it is likely that the model is going to see the same batch of data again and again
if we use multi threading and multiple workers
'''

import tensorflow as tf
import numpy as np


class SequenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, process_fn=None):
        """A `Sequence` implementation that can pre-process a mini-batch via `process_fn`

        Args:
            X: The numpy array of inputs.
            y: The numpy array of targets.
            batch_size: The generator mini-batch size.
            process_fn: The preprocessing function to apply on `X`
        """
        self.data = data
        self.batch_size = batch_size
        self.process_fn = process_fn

    def __len__(self):
        return len(self.X) // self.batch_size

    def on_epoch_end(self):
        pass

    def __getitem__(self, batch_idx):
        return self.process_fn(self.X, self.batch_size, batch_idx)


def batch_train(data, batch_size, batch_idx):
    number_rows = data.shape[0]
    predict_input = []
    y = []
    lstm_encoder_in = []
    lstm_decoder_in = []
    lstm_decoder_out = []
    i = batch_idx
    size = 0
    while size < batch_size:
        if data['week_no'].iloc[i] >= 35:
            i += 14
            continue
        if (data['Customer_id'].iloc[i] != data['Customer_id'].iloc[i + 12]) or (
                data['Product_id'].iloc[i] != data['Product_id'].iloc[i + 12]):
            i += 12
            if i >= number_rows - 12:
                i = 0
            continue
        lstm_encoder_in.append((data[['is_purchased']].iloc[i:i + 12]).values.tolist())
        temp = (data[['is_purchased']].iloc[i + 1:i + 13]).values.tolist()
        temp.reverse()
        lstm_decoder_out.append(temp)
        temp = data[['is_purchased']].iloc[i:i + 12].values.tolist()
        temp.reverse()
        lstm_decoder_in.append(temp)
        y.append(data['is_purchased'].iloc[i + 12])
        predict_input.append(list(data[['price', 'advertised']].iloc[i + 12]))
        i += 1
        size += 1
        if i >= number_rows - 12:
            i = 0
            size = batch_size
    return [np.array(lstm_encoder_in, dtype=np.float32), np.array(lstm_decoder_in, dtype=np.float32),
               np.array(predict_input, dtype=np.float32)], [np.array(lstm_decoder_out, dtype=np.float32),
                                                            np.array(lstm_decoder_out, dtype=np.float32),
                                                            np.array(y, dtype=np.float32)]


def batch_val(data, batch_size, batch_idx):
    number_rows = data.shape[0]
    predict_input = []
    y = []
    lstm_encoder_in = []
    lstm_decoder_in = []
    lstm_decoder_out = []
    i = batch_idx
    size = 0

    while size < batch_size:
        if data['week_no'].iloc[i] != 35:
            i += 1
            continue
        if (data['Customer_id'].iloc[i] != data['Customer_id'].iloc[i + 12]) or (
                data['Product_id'].iloc[i] != data['Product_id'].iloc[i + 12]):
            i += 12
            if i >= number_rows - 12:
                i = 0
            continue
        lstm_encoder_in.append((data[['is_purchased']].iloc[i:i + 12]).values.tolist())
        temp = (data[['is_purchased']].iloc[i + 1:i + 13]).values.tolist()
        temp.reverse()
        lstm_decoder_out.append(temp)
        temp = data[['is_purchased']].iloc[i:i + 12].values.tolist()
        temp.reverse()
        lstm_decoder_in.append(temp)
        y.append(data['is_purchased'].iloc[i + 12])
        predict_input.append(list(data[['price', 'advertised']].iloc[i + 12]))
        i += 49
        size += 1
        if i >= number_rows - 12:
            i = 0
            size = batch_size
    return [np.array(lstm_encoder_in, dtype=np.float32), np.array(lstm_decoder_in, dtype=np.float32),
           np.array(predict_input, dtype=np.float32)], [np.array(lstm_decoder_out, dtype=np.float32),
                                                            np.array(lstm_decoder_out, dtype=np.float32),
                                                            np.array(y, dtype=np.float32)]


def batch_test(data, batch_size, batch_idx):
    number_rows = data.shape[0]
    predict_input = []
    y = []
    lstm_encoder_in = []
    lstm_decoder_in = []
    lstm_decoder_out = []
    i = batch_idx
    size = 0

    while size < batch_size:
        if data['week_no'].iloc[i] != 36:
            i += 1
            continue
        if (data['Customer_id'].iloc[i] != data['Customer_id'].iloc[i + 12]) or (
                data['Product_id'].iloc[i] != data['Product_id'].iloc[i + 12]):
            i += 12
            if i >= number_rows - 12:
                i = 0
            continue
        lstm_encoder_in.append((data[['is_purchased']].iloc[i:i + 12]).values.tolist())
        temp = (data[['is_purchased']].iloc[i + 1:i + 13]).values.tolist()
        temp.reverse()
        lstm_decoder_out.append(temp)
        temp = data[['is_purchased']].iloc[i:i + 12].values.tolist()
        temp.reverse()
        lstm_decoder_in.append(temp)
        y.append(data['is_purchased'].iloc[i + 12])
        predict_input.append(list(data[['price', 'advertised']].iloc[i + 12]))
        i += 49
        size += 1
        if i >= number_rows - 12:
            i = 0
            size = batch_size
    return [np.array(lstm_encoder_in, dtype=np.float32), np.array(lstm_decoder_in, dtype=np.float32),
           np.array(predict_input, dtype=np.float32)], [np.array(lstm_decoder_out, dtype=np.float32),
                                                        np.array(lstm_decoder_out, dtype=np.float32),
                                                        np.array(y, dtype=np.float32)]
