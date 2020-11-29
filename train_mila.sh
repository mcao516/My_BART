module load anaconda/3
source activate py37

python train.py \
    --bart_path /home/mila/c/caomeng/Downloads/BART/bart.large.xsum.dummy/ \
    --checkpoint_file checkpoint1.pt \
    --data_name_or_path /home/mila/c/caomeng/Downloads/summarization/XSum/test_files/xsum-bin/ \
    --train_source /home/mila/c/caomeng/Downloads/summarization/XSum/test_files/train.bpe.source \
    --train_target /home/mila/c/caomeng/Downloads/summarization/XSum/test_files/train.bpe.target \
    --dev_source /home/mila/c/caomeng/Downloads/summarization/XSum/test_files/val.bpe.source \
    --dev_target /home/mila/c/caomeng/Downloads/summarization/XSum/test_files/val.bpe.target;