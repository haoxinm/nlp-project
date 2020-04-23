# nlp-project
GT CS7650 NLP project

This is the repository for the course project of CS7650 NLP. It contains our code for actor-critic text summarization network as well as 
our modified version of [Learning by Semantic Similarity Makes Abstractive Summarization Better](https://github.com/icml-2020-nlp/semsim).

## folder description
```
/--|fairseq/
   |short_story/
   |README.md
   |evaluate.py
   |network structure.png
```
*  `/fairseq` : The codes for our model. Modified from [fairseq (v 0.9.0)](https://github.com/pytorch/fairseq)
*  `/short_story` : The raw un-processed data for the English short story dataset we collect.
The `*.source` files contain the original documents and the `*.target` files contain the human annotated summaries.

Please notice that we currently don't provide any pre-trained models or processed data.
## requirements and installation
### using the model
#### prerequisites

- <b>python >= 3.6</b>
- <b>pytorch >= 1.4.0 (with CUDA)</b>
- <b>pytorch_transformers >= 1.2.0</b>

you also need to install our modified version of `fairseq`
```bash
cd fairseq
pip install --editable .
```

for faster model training, consider install the NVIDIA [apex](https://github.com/NVIDIA/apex) library.
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```


### dataset

When preparing the dataset we use, we follow the same procedure as the one used when pre-processing the CNN_DM dataset for BART.

#### 1) Download the CNN and Daily Mail data and preprocess it into data files with non-tokenized cased samples.

Follow the instructions [here](https://github.com/abisee/cnn-dailymail) to download the original CNN and Daily Mail datasets. To preprocess the data, refer to the pointers in [this issue](https://github.com/pytorch/fairseq/issues/1391) or check out the code [here](https://github.com/artmatsak/cnn-dailymail).

Follow the instructions [here](https://github.com/EdinburghNLP/XSum) to download the original Extreme Summarization datasets, or check out the code [here](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset), Please keep the raw dataset and make sure no tokenization nor BPE on the dataset.

#### 2) BPE preprocess:
Remember to change the `TASK` variable to the actual dataset used.
```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=cnn_dm
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
```

#### 3) Binarize dataset:
```bash
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```

The original `README` can be found [here](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md).

## fine-tune

If you wish to fine-tune the model yourself, please first download the `bart.large.cnn` checkpoint for actor, and `bart.large.mnli` checkpoint for critic [here](https://github.com/pytorch/fairseq/tree/master/examples/bart).

### 1. Prepare the dataset
### 2. Run fine-tuning
You can start fine-tuning by simply running the following shell script. Notice that you need to modify several parameters accordingly, 
such as `ACTOR_PATH` and `CRITIC_PATH` for where you store the checkpoints, `DEVICES` for which CUDA device to use,
`SEED` for which random seed to use, and `DATASET` for where you store the processed dataset.

Also, feel free to change other hyper-parameters for better performance.

```bash
TOTAL_NUM_UPDATES=1  

ACTOR_MAX_UPDATES=2048
ACTOR_MAX_EPOCH=100
ACTOR_WARMUP_UPDATES=39      
ACTOR_LR=3e-05                # Peak LR for polynomial LR scheduler.
ACTOR_UPDATE_FREQ=32
ACTOR_INTERVAL=512

CRITIC_MAX_UPDATES=256
CRITIC_MAX_EPOCH=1
CRITIC_WARMUP_UPDATES=13      
CRITIC_LR=5e-06                # Peak LR for polynomial LR scheduler.
CRITIC_UPDATE_FREQ=1
CRITIC_INTERVAL=256

NUM_CLASSES=2
MAX_SENTENCES=1        # Batch size.
MAX_TOKENS=1024
ACTOR_PATH=checkpoints/actor/checkpoint_best.pt
CRITIC_PATH=checkpoints/critic/checkpoint_best.pt

DEVICES=0
SEED=143
DATASET=cnn_dm-bin/

for I in $(seq 0 $((TOTAL_NUM_UPDATES-1)))
do
CUDA_VISIBLE_DEVICES=$DEVICES python3 fairseq/train.py $DATASET \
    --restore-file $CRITIC_PATH \
    --max-tokens $MAX_TOKENS\
    --max-sentences 1\
    --max-epoch 100 \
    --max-update $CRITIC_MAX_UPDATES \
    --layernorm-embedding \
    --share-all-embeddings \
    --seed $((SEED+I))\
    --share-decoder-input-output-embed \
    --fix-batches-to-gpus --ddp-backend "c10d" --all-gather-list-size 20480\
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-progress-bar --required-batch-size-multiple 1 \
    --task critic \
    --truncate-source \
    --curriculum -1 \
    --log-interval 64 \
    --source-lang source --target-lang target \
    --arch bart_large \
    --classification-head-name "critic" \
    --save-dir "checkpoints/critic" \
    --criterion ac_loss_critic \
    --actor-path "/home/marquess_mar96/nlp/checkpoints/actor" --actor-file "checkpoint_best.pt" \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $CRITIC_LR --total-num-update $CRITIC_MAX_UPDATES --warmup-updates $CRITIC_WARMUP_UPDATES \
    --fp16 --memory-efficient-fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --tensorboard-logdir "checkpoints/critic/log" \
    --update-freq $CRITIC_UPDATE_FREQ \
    --save-interval-updates $CRITIC_INTERVAL --keep-interval-updates 1 --patience 5 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state --no-save-criterion --validate-interval 100\
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;

CUDA_VISIBLE_DEVICES=$DEVICES python3 fairseq/train.py $DATASET \
    --restore-file $ACTOR_PATH \
    --max-tokens $MAX_TOKENS \
    --max-sentences 1\
    --max-epoch 100 \
    --max-update $ACTOR_MAX_UPDATES \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --curriculum -1 \
    --seed $((SEED+I))\
    --fix-batches-to-gpus --ddp-backend "c10d" --all-gather-list-size 20480\
    --log-interval 256 \
    --save-dir "checkpoints/actor" \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-progress-bar --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion ac_loss_actor\
    --critic-path "/home/marquess_mar96/nlp/checkpoints/critic" --critic-file "checkpoint_best.pt" \
    --critic-weight 50. --print-update 32768 --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --tensorboard-logdir "checkpoints/actor/log" \
    --lr-scheduler polynomial_decay --lr $ACTOR_LR --total-num-update $ACTOR_MAX_UPDATES --warmup-updates $ACTOR_WARMUP_UPDATES \
    --fp16 --memory-efficient-fp16 \
    --update-freq $ACTOR_UPDATE_FREQ \
    --save-interval-updates $ACTOR_INTERVAL --keep-interval-updates 1 --patience 5 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state --no-save-criterion --validate-interval 100\
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;  
done
``` 

Training for one epoch of `CNN_DM` dataset takes ~48 hrs on a single NVIDIA Tesla T4 GPU, and training with validation and checkpoint saving periodically takes longer time. We have trained the model for roughly 5 epochs to achieve the result in the final report.

## inference and evaluation

First you need to pre-process the dataset. Run following shell script to generate summaries.
```bash
DATASET=scisumm-corpus

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
  --checkpoint-path 'checkpoints/bart/scisumm' \
  --checkpoint-file 'checkpoint_best.pt' \
  --dataset $DATASET\
  --batch-size 50;
```

`--batch-size 50` works fine with 16GB VRAM GPU, you need to adjust this according to your hardware configuration. You may also check the original BART `README` for more details [here](https://github.com/icml-2020-nlp/semsim/blob/master/fairseq-semsim/examples/bart#evaluating-the-bartlargecnn-model).

Install [files2rouge](https://github.com/pltrdy/files2rouge) and [CoreNLP](https://stanfordnlp.github.io/CoreNLP/). Make sure you have added CoreNLP classpath to the environment variables.
Run the following shell script for evaluation `ROUGE` score.
```bash
cat $DATASET/test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $DATASET/test.hypo.tokenized
cat $DATASET/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $DATASET/test.hypo.target
files2rouge $DATASET/test.hypo.tokenized $DATASET/test.hypo.target
```
