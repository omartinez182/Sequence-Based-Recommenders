import pandas as pandas
import numpy as np
import torch
import gc
import os
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4RecF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, set_color
from recbole.utils.case_study import full_sort_topk
from recbole.data.interaction import Interaction


def main():
    """
    Sequence-based experiment RecSys with H&M Personalized Recommendations dataset.
    """
    # Item data
    df = pd.read_csv(r"data/articles.csv", dtype={'article_id': 'str'})
    df = df.drop(columns = ['product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
                            'perceived_colour_master_name', 'index_name', 'index_group_name', 'section_name', 
                            'garment_group_name', 'prod_name', 'department_name', 'detail_desc'])
    temp = df.rename(
        columns={'article_id': 'item_id:token', 'product_code': 'product_code:token', 'product_type_no': 'product_type_no:float',
                'product_group_name': 'product_group_name:token_seq', 'graphical_appearance_no': 'graphical_appearance_no:token', 
                'colour_group_code': 'colour_group_code:token', 'perceived_colour_value_id': 'perceived_colour_value_id:token', 
                'perceived_colour_master_id': 'perceived_colour_master_id:token', 'department_no': 'department_no:token', 
                'index_code': 'index_code:token', 'index_group_no': 'index_group_no:token', 'section_no': 'section_no:token', 
                'garment_group_no': 'garment_group_no:token'})

    temp.to_csv(r'data/kaggle/working/recbox_data/recbox_data.item', index=False, sep='\t')

    # Transaction data
    df = pd.read_csv(r"data/transactions_train.csv", dtype={'article_id': 'str'})
    df['t_dat'] = pd.to_datetime(df['t_dat'], format="%Y-%m-%d")
    df['timestamp'] = df.t_dat.values.astype(np.int64) // 10 ** 9
    temp = df[df['timestamp'] > 1585620000][['customer_id', 'article_id', 'timestamp']].rename(
        columns={'customer_id': 'user_id:token', 'article_id': 'item_id:token', 'timestamp': 'timestamp:float'})
    temp.to_csv('data/kaggle/working/recbox_data/recbox_data.inter', index=False, sep='\t')
    del temp
    gc.collect()

    # Default recs for user for which we can't produce predictions with the sequential model
    sub0 = pd.read_csv('data/input/submissio_byfone_chris.csv').sort_values('customer_id').reset_index(drop=True)
    sub1 = pd.read_csv('data/input/submission_trending.csv').sort_values('customer_id').reset_index(drop=True)
    sub2 = pd.read_csv('data/input/submission_exponential_decay.csv').sort_values('customer_id').reset_index(drop=True)
    sub0.columns = ['customer_id', 'prediction0']
    sub0['prediction1'] = sub1['prediction']
    sub0['prediction2'] = sub2['prediction']
    del sub1, sub2
    gc.collect()

    def cust_blend(dt, W = [1,1,1]):
        #Global ensemble weights
        #W = [1.15,0.95,0.85]
        
        #Create a list of all model predictions
        REC = []
        REC.append(dt['prediction0'].split())
        REC.append(dt['prediction1'].split())
        REC.append(dt['prediction2'].split())
        
        #Create a dictionary of items recommended. 
        #Assign a weight according the order of appearance and multiply by global weights
        res = {}
        for M in range(len(REC)):
            for n, v in enumerate(REC[M]):
                if v in res:
                    res[v] += (W[M]/(n+1))
                else:
                    res[v] = (W[M]/(n+1))
        
        # Sort dictionary by item weights
        res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
        
        # Return the top 12 itens only
        return ' '.join(res[:12])

    sub0['prediction'] = sub0.apply(cust_blend, W = [1.05,1.00,0.95], axis=1)
    del sub0['prediction0']
    del sub0['prediction1']
    del sub0['prediction2']
    gc.collect()
    sub0.to_csv(f'submission.csv', index=False)
    del sub0
    del df
    gc.collect()

    # Create dataset and train model with Recbole
    parameter_dict = {
        'data_path': 'data/kaggle/working',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'user_inter_num_interval': "[40,inf)",
        'item_inter_num_interval': "[40,inf)",
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp'],
                    'item': ['item_id', 'product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
                        'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                        'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
                },
        'selected_features': ['product_code', 'product_type_no', 'product_group_name', 'graphical_appearance_no',
                            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
                            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'],
        'neg_sampling': None,
        'epochs': 1,
        'eval_args': {
            'split': {'RS': [10, 0, 0]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'},
        'metrics': ['MAP'],
        'valid_metric': 'MAP@12'
    }

    config = Config(model='GRU4RecF', dataset='recbox_data', config_dict=parameter_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    # write config info into log
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = GRU4RecF(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {best_valid_result}')

    # GENERATE Predictions
    external_user_ids = dataset.id2token(
        dataset.uid_field, list(range(dataset.user_num)))[1:]#fist element in array is 'PAD'(default of Recbole) ->remove it 

    def add_last_item(old_interaction, last_item_id, max_len=50):
        new_seq_items = old_interaction['item_id_list'][-1]
        if old_interaction['item_length'][-1].item() < max_len:
            new_seq_items[old_interaction['item_length'][-1].item()] = last_item_id
        else:
            new_seq_items = torch.roll(new_seq_items, -1)
            new_seq_items[-1] = last_item_id
        return new_seq_items.view(1, len(new_seq_items))

    def predict_for_all_item(external_user_id, dataset, model):
        model.eval()
        with torch.no_grad():
            uid_series = dataset.token2id(dataset.uid_field, [external_user_id])
            index = np.isin(dataset.inter_feat[dataset.uid_field].numpy(), uid_series)
            input_interaction = dataset[index]
            test = {
                'item_id_list': add_last_item(input_interaction, 
                                            input_interaction['item_id'][-1].item(), model.max_seq_length),
                'item_length': torch.tensor(
                    [input_interaction['item_length'][-1].item() + 1
                    if input_interaction['item_length'][-1].item() < model.max_seq_length else model.max_seq_length])
            }
            new_inter = Interaction(test)
            new_inter = new_inter.to(config['device'])
            new_scores = model.full_sort_predict(new_inter)
            new_scores = new_scores.view(-1, test_data.dataset.item_num)
            new_scores[:, 0] = -np.inf  # set scores of [pad] to -inf
        return torch.topk(new_scores, 12)

    # Predictions
    predict_for_all_item('0109ad0b5a76924a1b58be677409bb601cc8bead9a87b8ce5b08a4a1f5bc71ef', 
                        dataset, model)

    # Top K items
    topk_items = []
    for external_user_id in external_user_ids:
        _, topk_iid_list = predict_for_all_item(external_user_id, dataset, model)
        last_topk_iid_list = topk_iid_list[-1]
        external_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
        topk_items.append(external_item_list)


if __name__ == "__main__":
    main()