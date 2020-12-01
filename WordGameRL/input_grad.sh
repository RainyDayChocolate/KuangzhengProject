#BERT=uncased_L-12_H-768_A-12
BERT=uncased_L-2_H-128_A-2

GAME_HOME=. \
python -u -m experiments.single.input_gradient \
--init_checkpoint ${BERT}/bert_model.ckpt \
--vocab_file ${BERT}/vocab.txt \
--bert_config_file ${BERT}/bert_config.json \
--output_dir test_bert_run \
--logger_file log.txt \
--learn_every 250 
