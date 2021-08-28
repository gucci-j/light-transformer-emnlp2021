Frustratingly Simple Pretraining Alternatives to Masked Language Modeling
===

This is the official implementation for "Frustratingly Simple Pretraining Alternatives to Masked Language Modeling" (EMNLP 2021).

## Requirements  
* torch
* transformers
* datasets
* scikit-learn
* tensorflow
* spacy
* matplotlib
* seaborn

## How to pre-train
### 1. Clone this repository
```
git clone https://github.com/gucci-j/light-transformer-emnlp2021.git
```

### 2. Install required packages  
```
cd ./light-transformer-emnlp2021
pip install -r requirements.txt
```  
> `requirements.txt` is located just under `light-transformer-emnlp2021`.

We also need spaCy's `en_core_web_sm` for preprocessing. If you have not installed this model, please run `python -m spacy download en_core_web_sm`.

### 3. Preprocess datasets  
```
cd ./src/utils
python preprocess_roberta.py --path=/path/to/save/data/
```  
You need to specify the following argument:
* `path`: (`str`) Where to save the processed data?  

### 4. Pre-training
You need to secify configs as command line arguments. Sample configs for pre-training MLM are shown as below. `python pretrainer.py --help` will display helper messages.  
```
cd ../
python pretrainer.py \
--data_dir=/path/to/dataset/ \
--do_train \
--learning_rate=1e-4 \
--weight_decay=0.01 \
--adam_epsilon=1e-8 \
--max_grad_norm=1.0 \
--num_train_epochs=1 \
--warmup_steps=12774 \
--save_steps=12774 \
--seed=42 \
--per_device_train_batch_size=16 \
--logging_steps=100 \
--output_dir=/path/to/save/weights/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=True \
--prediction_loss_only \
--fp16 \
--mlm_prob=0.15 \
--pretrain_model=RobertaForMaskedLM 
```
* `pretrain_model` should be selected from:   
    * `RobertaForMaskedLM` (MLM)  
    * `RobertaForShuffledWordClassification` (Shuffle)  
    * `RobertaForRandomWordClassification` (Random)  
    * `RobertaForShuffleRandomThreeWayClassification` (Shuffle+Random)  
    * `RobertaForFourWayTokenTypeClassification` (Token Type)  
    * `RobertaForFirstCharPrediction` (First Char)

#### Check the pre-training process  
You can monitor the progress of pre-training via the Tensorboard. Simply run the following:  
```
tensorboard --logdir=/path/to/log/dir/
```

#### Distributed training  
`pretrainer.py` is compatible with distributed training. Sample configs for pre-training MLM are as follows.  
```
python -m torch/distributed/launch.py \
--nproc_per_node=8 \
pretrainer.py \
--data_dir=/path/to/dataset/ \
--model_path=None \
--do_train \
--learning_rate=5e-5 \
--weight_decay=0.01 \
--adam_epsilon=1e-8 \
--max_grad_norm=1.0 \
--num_train_epochs=1 \
--warmup_steps=24000 \
--save_steps=1000 \
--seed=42 \
--per_device_train_batch_size=8 \
--logging_steps=100 \
--output_dir=/path/to/save/weights/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm \
--prediction_loss_only \
--fp16 \
--mlm_prob=0.15 \
--pretrain_model=RobertaForMaskedLM 
```
> For more details about `launch.py`, please refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.


#### Mixed precision training  
**Installation**
* For PyTorch version >= 1.6, there is [a native functionality](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) to enable mixed precision training.  
* For older versions, [NVIDIA apex](https://github.com/NVIDIA/apex) must be installed.  
    > * You might encounter some errors when installing `apex` due to permission problems. To fix these, specify `export TMPDIR='/path/to/your/favourite/dir/'` and change permissions of all files under `apex/.git/` to 777.  
    > * You also need to specify an optimisation method from [https://nvidia.github.io/apex/amp.html](https://nvidia.github.io/apex/amp.html).  

**Usage**  
To use mixed precision during pre-training, just specify `--fp16` as an input argument. For older PyTorch versions, also specify `--fp16_opt_level` from `O0`, `O1`, `O2`, and `O3`.  


## How to fine-tune  
### GLUE  
1. **Download GLUE data**  
    ```
    git clone https://github.com/huggingface/transformers
    python transformers/utils/download_glue_data.py
    ```

2. **Create a json config file**  
You need to create a `.json` file for configuration or use command line arguments. A sample `.json` file is available in `./params/finetune/glue/`.  

    ```json
    {
        "model_name_or_path": "/path/to/pretrained/weights/",
        "tokenizer_name": "roberta-base",
        "task_name": "MNLI",
        "do_train": true,
        "do_eval": true,
        "data_dir": "/path/to/MNLI/dataset/",
        "max_seq_length": 128,
        "learning_rate": 2e-5,
        "num_train_epochs": 3, 
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 128,
        "logging_steps": 500,
        "logging_first_step": true,
        "save_steps": 1000,
        "save_total_limit": 2,
        "evaluate_during_training": true,
        "output_dir": "/path/to/save/models/",
        "overwrite_output_dir": true,
        "logging_dir": "/path/to/save/log/files/",
        "disable_tqdm": true
    }
    ```
    > For `task_name` and `data_dir`, please choose one from CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, and WNLI.  

3. **Fine-tune**  
    ```
    python run_glue.py /path/to/json/
    ```
    > Instead of specifying a JSON path, you can directly specify configs as input arguments.  
    > You can also monitor training via Tensorboard.   
    > `--help` option will display a helper message. 


### SQuAD  
1. **Download SQuAD data**  
    ```
    cd ./utils
    python download_squad_data.py --save_dir=/path/to/squad/
    ```

2. **Fine-tune**  
    ```
    cd ..
    export SQUAD_DIR=/path/to/squad/
    python run_squad.py \
    --model_type roberta \
    --model_name_or_path=/path/to/pretrained/weights/ \
    --tokenizer_name roberta-base \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir=$SQUAD_DIR \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay=0.01 \
    --warmup_steps=3327 \
    --num_train_epochs 10.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps=278 \
    --save_steps=50000 \
    --patience=5 \
    --objective_type=maximize \
    --metric_name=f1 \
    --overwrite_output_dir \
    --evaluate_during_training \
    --output_dir=/path/to/save/weights/ \
    --logging_dir=/path/to/save/logs/ \
    --seed=42 
    ```
    > Similar to pre-training, you can monitor the fine-tuning status via Tensorboard.   
    > `--help` option will display a helper message. 


## Citation  
```
@inproceedings{yamaguchi-etal-2021-frustratingly,
    title = "Frustratingly Simple Pretraining Alternatives to Masked Language Modeling",
    author = "Yamaguchi, Atsuki  and
      Chrysostomou, George  and
      Margatina, Katerina  and
      Aletras, Nikolaos",
    booktitle = "Proceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## License
[MIT License](./LICENSE)