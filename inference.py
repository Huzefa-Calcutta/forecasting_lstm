import pandas as pd
import datetime
import sys
import configparser
import os
import itertools
import tensorflow as tf
import numpy as np





def preprocess(future_data_path, hist_data_path, num_time_steps, future_start_week, list_customers,
               list_products):
    future_data = pd.read_csv(future_data_path).rename(
        columns={"i": "Customer_id", "j": "Product_id"})
    future_data['week_no'] = future_start_week
    hist_data = pd.read_csv(hist_data_path).rename(columns={"i": "Customer_id", "j": "Product_id", "t": "week_no"})
    hist_data['is_purchased'] = 1
    last_n_time_steps = sorted(pd.unique(hist_data['week_no']), reverse=True)[:12]
    for i in range(future_start_week - num_time_steps, future_start_week, 1):
        if i not in last_n_time_steps:
            raise ValueError("The previous historical data must contain atleast 12 weeks before the current week")
    weeks = list(range(future_start_week - num_time_steps, future_start_week))
    # data of the last week just before the future week

    # price for future weeks is price_last_week*discount
    last_1_time_step_prod_price_data = pd.DataFrame(list(itertools.product(*[list_products, [future_start_week - 1]])),
                                                    columns=['Product_id', 'week_no'])

    last_1_time_step_prod_price_data = pd.merge(last_1_time_step_prod_price_data,
                                                hist_data[['Product_id', 'week_no', 'price']].drop_duplicates(),
                                                how='left',
                                                on=['Product_id', 'week_no'])

    # Future price requires price for week just before the future week. if price not available, impute from past data

    for product in pd.unique(
            last_1_time_step_prod_price_data[last_1_time_step_prod_price_data['price'].isna()]['Product_id']):
        for week in pd.unique(
                last_1_time_step_prod_price_data[last_1_time_step_prod_price_data['price'].isna()]['week_no']):
            impute_value = pd.unique(
                hist_data.loc[(hist_data['Product_id'] == product) & (hist_data['week_no'] == week - 1), 'price'])[0]
            last_1_time_step_prod_price_data.loc[(last_1_time_step_prod_price_data['price'].isna())
                                                 & (last_1_time_step_prod_price_data['Product_id'] == product)
                                                 & (last_1_time_step_prod_price_data[
                                                        'week_no'] == week), 'price'] = impute_value

    future_prod_data = pd.DataFrame(list(itertools.product(*[list_products, [future_start_week]])), columns=[
        'Product_id', 'week_no'])

    future_prod_data = pd.merge(future_prod_data, future_data[['Product_id', 'week_no', 'discount']].drop_duplicates(),
                                how='outer', on=['Product_id', 'week_no']).fillna(
        0)  # if no discount value present we assume discount is 0 % for that product and week

    future_prod_data = pd.merge(future_prod_data,
                                future_data[['Product_id', 'week_no', 'advertised']].drop_duplicates(), how='outer',
                                on=['Product_id', 'week_no']).fillna(
        0)  # if no value for given week and product_id, assume no advertisement for that product and week
    # getting the price coulmn

    future_prod_data = pd.merge(future_prod_data,
                                last_1_time_step_prod_price_data[['Product_id', 'price']].drop_duplicates(), how='left',
                                on=['Product_id'])

    future_prod_data['price'] = future_prod_data['price'] * (1 - future_prod_data['discount'])

    # complete data with all customer id and product ids and weeks combination:
    hist_pred_data_df = pd.DataFrame(list(itertools.product(*[list_customers, list_products, weeks])),
                                     columns=['Customer_id', 'Product_id', 'week_no'])
    hist_pred_data_df = pd.merge(hist_pred_data_df,
                                 hist_data,
                                 how='left', on=['Customer_id', 'Product_id', 'week_no']).fillna(0)

    future_pred_data = pd.DataFrame(list(itertools.product(*[list_customers, list_products, [future_start_week]])),
                                    columns=['Customer_id', 'Product_id', 'week_no'])

    future_pred_data = pd.merge(future_pred_data, future_prod_data, on=['Product_id', 'week_no'])
    future_pred_data = future_pred_data.drop(columns=['discount'])
    future_pred_data = hist_pred_data_df.append(future_pred_data, ignore_index=True)
    future_pred_data = future_pred_data[future_pred_data['week_no'] >= future_start_week - 12]
    future_pred_data = future_pred_data.sort_values(by=['Customer_id', 'Product_id', 'week_no']).reset_index()

    return future_pred_data


def prediction(future_pred_data, model, pred_week):
    lstm_encoder_in = []
    lstm_decoder_in = []
    predict_input = []
    i = 0
    while i < future_pred_data.shape[0] - 12:
        if (future_pred_data['Customer_id'].iloc[i] != future_pred_data['Customer_id'].iloc[i + 12]) or (
                future_pred_data['Product_id'].iloc[i] != future_pred_data['Product_id'].iloc[i + 12]):
            i += 12
            continue
        lstm_encoder_in.append(future_pred_data[['is_purchased']].iloc[i:i + 12].values.tolist())
        temp = future_pred_data[['is_purchased']].iloc[i:i + 12].values.tolist()
        temp.reverse()
        lstm_decoder_in.append(temp)
        predict_input.append(list(future_pred_data[['price', 'advertised']].iloc[i + 12]))
        i += 1
    print("start model inference")
    prediction = model.predict([np.array(lstm_encoder_in, dtype=np.float32),
                                np.array(lstm_decoder_in, dtype=np.float32),
                                np.array(predict_input, dtype=np.float32)])
    return prediction[2]


def main(future_data_path, hist_data_path, num_time_steps, future_start_week, list_customers, list_products, model_path,
         prediction_dir):
    future_pred_data = preprocess(future_data_path, hist_data_path, num_time_steps, future_start_week, list_customers,
                                  list_products)
    # loading the model
    model = tf.keras.models.load_model(model_path)

    future_pred_data['is_purchased'] = prediction(future_pred_data, model, future_start_week)
    future_pred_data[['Customer_id', 'Product_id','week_no', 'is_purchased']].to_csv(os.path.join(prediction_dir, "prediction.csv"), index=False)


if __name__ == '__main__':
    # loading the config file
    cfgParse = configparser.ConfigParser()
    cfgParse.read(sys.argv[1])

    future_file = cfgParse.get("data", "future_file")
    hist_file = cfgParse.get("data", "hist_file")
    num_time_steps = cfgParse.get("data", "num_time_steps")
    future_start_week = cfgParse.get("data", "future_start_week")
    customers = list(range(2000))
    products = list(range(40))

    model_loc = cfgParse.get("model", "model_loc")
    prediction_dir = cfgParse.get('output', 'predicted_dir')

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

        
    prediction_time_st = datetime.datetime.now()
    main(future_file, hist_file, num_time_steps, future_start_week, customers, products,model_loc,prediction_dir)
    prediction_time_end = datetime.datetime.now()
    prediction_time = (prediction_time_end - prediction_time_st).total_seconds() / 60.0
    print("Time required for prediction for random forest model is %.3f minutes" % prediction_time)
