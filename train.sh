python train.py \
    --bart_path /home/ml/cadencao/fairseq/checkpoints/dummy_checkpoint \
    --checkpoint_file checkpoint1.pt \
    --data_name_or_path /home/ml/cadencao/XSum/test_files/xsum-bin/ \
    --train_source /home/ml/cadencao/XSum/test_files/train.bpe.source \
    --train_target /home/ml/cadencao/XSum/test_files/train.bpe.target \
    --dev_source /home/ml/cadencao/XSum/test_files/val.bpe.source \
    --dev_target /home/ml/cadencao/XSum/test_files/val.bpe.target;