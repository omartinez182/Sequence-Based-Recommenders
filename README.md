
# Sequence & Session-Based Recommendations
# RecBole

[HomePage]: https://recbole.io/

In this repository we evaluate two models with 2 different datasets using the RecBole framework, which is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified, comprehensive and efficient framework.


## Installation

```bash
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole.py
```

This script will run the BPR model on the ml-100k dataset.

Typically, this example takes less than one minute. We will obtain some output like:

```
INFO ml-100k
The number of users: 944
Average actions of users: 106.04453870625663
The number of items: 1683
Average actions of items: 59.45303210463734
The number of inters: 100000
The sparsity of the dataset: 93.70575143257098%

INFO Evaluation Settings:
Group by user_id
Ordering: {'strategy': 'shuffle'}
Splitting: {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
Negative Sampling: {'strategy': 'full', 'distribution': 'uniform'}

INFO BPRMF(
    (user_embedding): Embedding(944, 64)
    (item_embedding): Embedding(1683, 64)
    (loss): BPRLoss()
)
Trainable parameters: 168128

INFO epoch 0 training [time: 0.27s, train loss: 27.7231]
INFO epoch 0 evaluating [time: 0.12s, valid_score: 0.021900]
INFO valid result:
recall@10: 0.0073  mrr@10: 0.0219  ndcg@10: 0.0093  hit@10: 0.0795  precision@10: 0.0088

...

INFO epoch 63 training [time: 0.19s, train loss: 4.7660]
INFO epoch 63 evaluating [time: 0.08s, valid_score: 0.394500]
INFO valid result:
recall@10: 0.2156  mrr@10: 0.3945  ndcg@10: 0.2332  hit@10: 0.7593  precision@10: 0.1591

INFO Finished training, best eval result in epoch 52
INFO Loading model structure and parameters from saved/***.pth
INFO best valid result:
recall@10: 0.2169  mrr@10: 0.4005  ndcg@10: 0.235  hit@10: 0.7582  precision@10: 0.1598
INFO test result:
recall@10: 0.2368  mrr@10: 0.4519  ndcg@10: 0.2768  hit@10: 0.7614  precision@10: 0.1901
```

If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```bash
python run_recbole.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run_recbole.py --model=[model_name]
```

### Auto-tuning Hyperparameter 
Open `RecBole/hyper.test` and set several hyperparameters to auto-searching in parameter list. The following has two ways to search best hyperparameter:
* **loguniform**: indicates that the parameters obey the uniform distribution, randomly taking values from e^{-8} to e^{0}.
* **choice**: indicates that the parameter takes discrete values from the setting list.

Here is an example for `hyper.test`: 
```
learning_rate loguniform -8, 0
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
mlp_hidden_size choice ['[64, 64, 64]','[128, 128]']
```
Set training command parameters as you need to run:
```
python run_hyper.py --model=[model_name] --dataset=[data_name] --config_files=xxxx.yaml --params_file=hyper.test
e.g.
python run_hyper.py --model=BPR --dataset=ml-100k --config_files=test.yaml --params_file=hyper.test
```
Note that `--config_files=test.yaml` is optional, if you don't have any customize config settings, this parameter can be empty.

This processing maybe take a long time to output best hyperparameter and result:
```
running parameters:                                                                                                                    
{'embedding_size': 64, 'learning_rate': 0.005947474154838498, 'mlp_hidden_size': '[64,64,64]', 'train_batch_size': 512}                
  0%|                                                                                           | 0/18 [00:00<?, ?trial/s, best loss=?]
```

More information about parameter tuning can be found in our [docs](https://recbole.io/docs/user_guide/usage/parameter_tuning.html).


## Time and Memory Costs
We constructed preliminary experiments to test the time and memory cost on three different-sized datasets 
(small, medium and large). For detailed information, you can click the following links.

* [General recommendation models](asset/time_test_result/General_recommendation.md)
* [Sequential recommendation models](asset/time_test_result/Sequential_recommendation.md)
* [Context-aware recommendation models](asset/time_test_result/Context-aware_recommendation.md)
* [Knowledge-based recommendation models](asset/time_test_result/Knowledge-based_recommendation.md)

NOTE: Our test results only gave the approximate time and memory cost of our implementations in the RecBole library
(based on our machine server).  Any feedback or suggestions about the implementations and test are welcome. 
We will keep improving our implementations, and update these test results.


## RecBole Major Releases
| Releases  | Date   |
|-----------|--------|
| v1.0.0    | 09/17/2021 |
| v0.2.0    | 01/15/2021 |
| v0.1.1    | 11/03/2020 |

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/RecBole/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

We thank the insightful suggestions from [@tszumowski](https://github.com/tszumowski), [@rowedenny](https://github.com/rowedenny), [@deklanw](https://github.com/deklanw) et.al.

We thank the nice contributions through PRs from [@rowedenny](https://github.com/rowedenny)，[@deklanw](https://github.com/deklanw) et.al.

## Cite
If you find RecBole useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2011.01731):

```
@article{recbole,
    title={RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
    author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
    year={2020},
    journal={arXiv preprint arXiv:2011.01731}
}
```

## The Team
RecBole is developed and maintained by [RUC, BUPT, ECNU](https://www.recbole.io/about.html).

Here is the list of our lead developers in each development phase. They are the souls of RecBole and have made outstanding contributions.

|         Time          |        Version         |                Lead Developers                 |
| :-------------------: | :--------------------: | :--------------------------------------------: |
| June 2020<br> ~<br> Nov. 2020 |        v0.1.1         |  Shanlei Mu ([@ShanleiMu](https://github.com/ShanleiMu)), Yupeng Hou ([@hyp1231](https://github.com/hyp1231)),<br> Zihan Lin ([@linzihan-backforward](https://github.com/linzihan-backforward)), Kaiyuan Li ([@tsotfsk](https://github.com/tsotfsk))|
|    Nov. 2020<br> ~ <br> now    |  v0.1.2 ~ v1.0.0 |      Yushuo Chen ([@chenyushuo](https://github.com/https://github.com/chenyushuo)), Xingyu Pan ([@2017pxy](https://github.com/2017pxy))    |


## License
RecBole uses [MIT License](./LICENSE).