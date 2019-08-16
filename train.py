#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import itertools
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, concatenate, RepeatVector, Reshape, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from sklearn.metrics import roc_auc_score
import configparser
import sys
from utils_data import *

# setting the float data type
backend.set_floatx('float32')
backend.set_epsilon(1e-7)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

cfgParse = configparser.ConfigParser()
cfgParse.read(sys.argv[1])

train_data_path = cfgParse.get("data", "train_data")
output_dir = cfgParse.get("output", "model_dir")

data = pd.read_csv(train_data_path).rename(columns={"i": "Customer_id", "j": "Product_id", "t": "week_no"})
data['is_purchased'] = 1
weeks = list(range(49))
customers = list(range(2000))
products = list(range(40))
data_unstacked = pd.DataFrame(list(itertools.product(*[customers, products, weeks])),
                              columns=['Customer_id', 'Product_id', 'week_no'])

data_unstacked = pd.merge(data_unstacked, data[['Product_id', 'week_no', 'price', 'advertised']].drop_duplicates(),
                          how='outer', on=['Product_id', 'week_no'])
data_unstacked = pd.merge(data_unstacked, data[['Customer_id', 'Product_id', 'week_no', 'is_purchased']], how='left',
                          on=['Customer_id', 'Product_id', 'week_no'])

data_unstacked['advertised'] = data_unstacked['advertised'].fillna(0)
data_unstacked['is_purchased'] = data_unstacked['is_purchased'].fillna(0)

for product in pd.unique(data_unstacked[data_unstacked['price'].isna()]['Product_id']):
    for week in pd.unique(data_unstacked[data_unstacked['price'].isna()]['week_no']):
        impute_value = pd.unique(data.loc[(data['Product_id'] == product) & (data['week_no'] == week - 1), 'price'])[0]
        data_unstacked.loc[(data_unstacked['price'].isna()) & (data_unstacked['Product_id'] == product) & (
                    data_unstacked['week_no'] == week), 'price'] = impute_value

data_unstacked = data_unstacked.sort_values(by=['Customer_id', 'Product_id', 'week_no']).reset_index()


def auc_roc(y_true, y_pred):
    return tf.py_function(
        lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float16'),
        [y_true, y_pred],
        'float32',
        name='sklearnAUC')


def build_lstm_encoder_model(num_lstm_layers, lstm_neuron, num_dense_layers, dense_neuron, input_features,
                             input_sequence_size):
    if isinstance(lstm_neuron, int):
        lstm_neuron_list = [lstm_neuron] * num_lstm_layers
    if isinstance(lstm_neuron, (list, tuple, np.ndarray, set)):
        if len(lstm_neuron) < num_lstm_layers:
            raise ValueError(
                "The number of neurons for all lstm layers have not been specified. Please ensure the lstm_neuron has same numebr of elements as num_lstm_layers")
        lstm_neuron_list = lstm_neuron

    if isinstance(dense_neuron, int):
        dense_neuron_list = [dense_neuron] * num_dense_layers
    if isinstance(dense_neuron, (list, tuple, np.ndarray, set)):
        if len(dense_neuron) < num_dense_layers:
            raise ValueError(
                "The number of neurons for all dense layers have not been specified. Please ensure the dense_neuron has same number of elements as num_dense_layers")
        dense_neuron_list = dense_neuron

    lstm_inp_layer = Input(shape=(input_sequence_size, 1))
    encoded = lstm_inp_layer
    for i in range(num_lstm_layers - 1):
        return_seq = True
        encoding_lstm = LSTM(lstm_neuron_list[i], return_sequences=return_seq, dropout=0.4, recurrent_dropout=0.4,
                             activation=None, kernel_initializer='glorot_uniform', name="lstm_enc_%d" % i)
        encoded = encoding_lstm(encoded)
        encoded = BatchNormalization()(encoded)
        enocded = Activation('tanh')

    # last encoder layer
    encoding_lstm = LSTM(lstm_neuron_list[-1], return_sequences=False, return_state=True,
                         kernel_initializer='glorot_uniform', dropout=0.4, recurrent_dropout=0.4)
    encoded, encoded_hidden_state, encoded_cell_state = encoding_lstm(
        encoded)  # getting cell state of last encoder layer
    decoded_conditioned_input = Input(shape=(input_sequence_size, 1))
    decoded_conditioned = decoded_conditioned_input
    decoded = RepeatVector(input_sequence_size)(encoded)

    # Decoder
    for i in range(num_lstm_layers):
        decoding_lstm = LSTM(lstm_neuron_list[-1 - i], return_sequences=True, activation=None,
                             kernel_initializer='glorot_uniform', name="lstm_dec_%d" % i)
        decoded = decoding_lstm(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Activation('tanh')(decoded)

    # Decoder conditioned on current time step data input
    for i in range(num_lstm_layers):
        decoded_conditioned = BatchNormalization()(decoded_conditioned)
        if i == 0:
            decoding_lstm = LSTM(lstm_neuron_list[-1 - i], return_sequences=True, kernel_initializer='glorot_uniform',
                                 name="lstm_dec_cond_%d" % i)
            decoded_conditioned = decoding_lstm(decoded_conditioned,
                                                initial_state=[encoded_hidden_state, encoded_cell_state])
        else:
            decoding_lstm = LSTM(lstm_neuron_list[-1 - i], return_sequences=True, kernel_initializer='glorot_uniform',
                                 name="lstm_dec_cond_%d" % i)
            decoded_conditioned = decoding_lstm(decoded_conditioned)

    decoded = TimeDistributed(Dense(1), name='decoder_output')(decoded)
    decoded_conditioned_output = TimeDistributed(Dense(1), name='conitioned_decoder_output')(decoded_conditioned)

    feat_input = Input(shape=(input_features,))
    dense_inp = concatenate([encoded_cell_state, feat_input])

    for i in range(num_dense_layers):
        dense_inp = Dropout(0.4)(dense_inp)
        dense_inp = Dense(dense_neuron_list[i], name="dense_layer_%d" % i)(dense_inp)
        dense_inp = BatchNormalization()(dense_inp)
        dense_inp = Activation('sigmoid')(dense_inp)

    output = Dense(1, activation='sigmoid', name="final_output")(dense_inp)
    model = Model(inputs=[lstm_inp_layer, decoded_conditioned_input, feat_input],
                  outputs=[decoded, decoded_conditioned_output, output])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


model_products = build_lstm_encoder_model(3, [256, 64, 32], 3, 64, 2, 12)
# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 16} )
# sess = tf.Session(config=config)
# tf.keras.backend.set_session(sess)

earlystop = EarlyStopping(monitor='val_final_output_loss', patience=20, mode='min', restore_best_weights=True) # ensures that we have model weights corresponding to the best value of the metric at the end of

# make tensorbaord log_dir
if not os.path.exists("logs"):
    os.mkdir("logs")
tensorboard = TensorBoard(log_dir='./logs', write_graph=True, update_freq='epoch')

# Saving model_checkpoint
filepath = "forecaster_model-{epoch:02d}-{val_final_output_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_final_output_loss', verbose=1,
                             save_best_only=True,
                             mode='min', save_weights_only=False)
with tf.device('/gpu:2'):
    history = model_products.fit_generator(SequenceDataGenerator(data_unstacked, 500, batch_train), epochs=500,
                                           steps_per_epoch=int(data_unstacked.shape[0] / 500) + 1,
                                           validation_data=SequenceDataGenerator(data_unstacked, 500, batch_val),
                                           validation_steps=int(data_unstacked.shape[0] / 500) + 1,
                                           callbacks=[checkpoint, earlystop, tensorboard], verbose=2, shuffle=False,
                                           use_multiprocessing=True,
                                           workers=6)



model_products.save(os.path.join(output_dir, "best_model.h5"))

# calculating auc on test data
_, _, prediction = model_products.predict_generator(SequenceDataGenerator(data_unstacked, 500, batch_test), steps=80000/500, verbose=1, workers=1)

test_auc = roc_auc_score(data_unstacked[data_unstacked['week_no'] == 48]['is_purchased'], prediction, average='micro')
print("The auc value of the model is %.3f" % test_auc)
