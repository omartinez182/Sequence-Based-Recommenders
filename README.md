
# Session-Based Recommendations
# RecBole

[HomePage]: https://recbole.io/

In this repository we evaluate two models with 2 different datasets using the RecBole framework, which is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified, comprehensive and efficient framework.

This repository was forked from the [RecBole](https://github.com/RUCAIBox/RecBole) GitHub repository, and our main contribution is the script for the H&M Dataset experimentation and some slight modifications to the session-based recommender for the DIGINETICA dataset for demonstration/benchmarking purposes.


## Installation

```bash
git clone https://github.com/omartinez182/Sequence-Based-Recommenders.git && cd Sequence-Based-Recommenders
pip install -e . --verbose
```


### Diginetica Experiment
To run the first experiment you can use the following command:

```bash
python3 run_example/session_based_rec_experiment.py
```

This script will run the GRU4Rec experiment on the Diginetica dataset.

If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```bash
python run_example/session_based_rec_experiment.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run_example/session_based_rec_experiment.py --model=[model_name]
```

Benchmarks for the Diginetica dataset can be found on the [CIKM Cup 2016 Track 2: Personalized E-Commerce Search Challenge](https://competitions.codalab.org/competitions/11161#learn_the_details-evaluation) under the evaluation tab and the related work sections, an example would be the paper [Improving One-Class Collaborative Filtering by Incorporating Rich User Information](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.7135&rep=rep1&type=pdf).


### H&M Personalized Recommendations Experiment

In order to run this experiment, first you must download the necessary data from Kaggle, you can connect to the API by running the following command:

```bash
pip install kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
Then you can proceed to download the data (NOTE that this is over 32GB) and unzip it.

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
kaggle datasets download -d astrung/hm-pre-recommendation
unzip h-and-m-personalized-fashion-recommendations.zip
unzip hm-pre-recommendation.zip
```
Finally, you can run the experiment with:

```bash
python run_example/h_m_rec_experiment.py
```


## Citations
[H&M](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/code?competitionId=31254&searchQuery=sequence) & 
[RecBole](https://arxiv.org/abs/2011.01731):
```
@article{recbole,
    title={RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
    author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
    year={2020},
    journal={arXiv preprint arXiv:2011.01731}
}
```
