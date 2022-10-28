# Multi-Document Scientific Summarization from a Knowledge Graph-Centric View
This is the Pytorch implementation for [Multi-Document Scientific Summarization from a Knowledge Graph-Centric View], accepted by COLING 2022.

<p align="center">
 <img src="images/model_arc.png" width="700"/>
</p>

## Requirements
* Python == 3.6.3
* Pytorch == 1.5.0
* transformers == 4.10.3
* dgl-cu101 == 0.6.1
* pyrouge == 0.1.3
* rake-nltk

## Usage
1. Create folder `datasets`, `cache`, `trained_model`, `result` , `log` under the root directory.

2. Download Multi-Xscience Dataset from [here](https://github.com/yaolu/Multi-XScience).

3. Train a Dygie++ model for extracting entitie and relations from [here](https://github.com/dwadden/dygiepp).

4. Dataset Format and Dataset Preprocessing:
    * 4.1 The format of the input files are shown in folder `example_data` , including `**.label.jsonl`, `**.ent_type_relation.jsonl`,`**.ent_promptsummary.jsonl`, `**.ent_importance_score.jsonl`, `**_summary.ent_importance_score.jsonl` and `output_**_processed_coref.json`.
    
    * 4.2 Extract entities and relations for Multi-Xscience using Dygie++. The output file is `output_**_processed_coref.json`.
    Postprocess the output file `output_**_processed_coref.json` to the format of `**.ent_type_relation.jsonl`
    
    * 4.3 Using RAKE algorithm on `output_**_processed_coref.json` to calculate RAKE score for each entity candidate. 
    ```
    python script/keyphrase_extract.py
    ```
    
    * 4.4 Create KGText samples using `**_summary.ent_importance_score.jsonl`.
    ```
    python script/add_prompt_info.py
    ```

## Training a new model
```bash
export PYTHONPATH=.

CUDA_LAUNCH_BLOCKING=1 python train.py  --mode train --cuda  --data_dir <path-to-datasets-folder> --batch_size 4 --seed 666 --train_steps 100000 --save_checkpoint_steps 4000  --report_every 1  --visible_gpus 0 --gpu_ranks 0  --world_size 1 --accum_count 2 --dec_dropout 0.1 --enc_dropout 0.1  --model_path  <path-to-trained-model-folder>  --log_file <path-to-log-file>  --inter_layers 6,7 --inter_heads 8 --hier --doc_max_timesteps 50 --prop 3 --min_length1 100 --no_repeat_ngram_size1 3 --sep_optim false --num_workers 5 --lr_dec 0.05 --warmup_steps 8000 --lr 0.005 --enc_layers 6  --dec_layers 6 --use_nucleus_sampling false --label_smoothing 0.1 --entloss_weight 1 
```

## Test
```bash
export PYTHONPATH=.

python train.py  --mode test --cuda  --data_dir <path-to-datasets-folder> --batch_size 8 --valid_batch_size 8 --seed 666   --visible_gpus 0 --gpu_ranks 0 --dec_dropout 0.1 --enc_dropout 0.1  --lr 0.2 --label_smoothing 0.0  --log_file <path-to-log-file>  --inter_layers 6,7 --inter_heads 8 --doc_max_timesteps 50 --use_bert false --report_rouge --alpha 0.4 --max_length 400 --result_path <path-to-result-folder> --prop 3 --test_all false --sep_optim false   --use_bert false  --use_nucleus_sampling false --min_length1 100 --min_length2 110 --no_repeat_ngram_size1 3 --no_repeat_ngram_size2 3 --test_from <path-to-saved-model-checkpoint>
```

## References
```
@inproceedings{wang2022multi,
  title={Multi-Document Scientific Summarization from a Knowledge Graph-Centric View},
  author={Wang, Pancheng and Li, Shasha and Pang, Kunyuan and He, Liangliang and Li, Dong and Tang, Jintao and Wang, Ting},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={6222--6233},
  year={2022}
}
```