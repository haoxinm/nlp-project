#INPUT_COUNT=2
#SPLITS="val_quarter"
#TASK_DATA_FOLDER="scisumm-corpus"
#
#for SPLIT in $SPLITS
#do
#for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
#do
#  LANG="input$INPUT_TYPE"
#  echo "BPE encoding $SPLIT/$LANG"
#  python3 -m examples.roberta.multiprocessing_bpe_encoder \
#  --encoder-json encoder.json \
#  --vocab-bpe vocab.bpe \
#  --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$LANG" \
#  --outputs "$TASK_DATA_FOLDER/processed/$SPLIT.$LANG" \
#  --workers 60 \
#  --keep-empty;
#done
#done
#
#for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
#do
#  LANG="input$INPUT_TYPE"
#  fairseq-preprocess \
#    --only-source \
#    --trainpref "$TASK_DATA_FOLDER/processed/train.$LANG" \
#    --validpref "$TASK_DATA_FOLDER/processed/val.$LANG" \
#    --destdir "$TASK_DATA_FOLDER-bin/$LANG" \
#    --workers 60 \
#    --srcdict dict.txt;
#done
#
#fairseq-preprocess \
#      --only-source \
#      --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
#      --validpref "$TASK_DATA_FOLDER/processed/val.label" \
#      --destdir "$TASK_DATA_FOLDER-bin/label" \
#      --workers 60;



#TASK=scisumm-corpus
TASK=short_story
for SPLIT in train val
do
  for LANG in source target
  do
    python fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done


fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;