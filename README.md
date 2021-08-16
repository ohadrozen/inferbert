# inferbert
Dataset as described on our paper "Teach the Rules, Provide the Facts: Targeted Relational-knowledge Enhancement for Textual Inference"

Examples:
1)
Train Inferbert on our 4 phenomena training set and test it on the hypernymy_unseen set, while adding 10K MNLI examples, trying 20 different initializations of S-KAR and scanning 2 different learning rates:

python main.py \
--train_setname train_four_phenomena \
--dev_setname dev_four_phenomena \
--test_setname hypernymy_test_unseen \
--mix_teaching_with_MNLI \
--max_mix_MNLI_examples 10000 \
--learning_rate_vec 2e-5 6e-5 \
--num_of_rand_init 20 \



2) loading the trained model ‘2’ and testing it on hypernymy_test_seen
python main.py \
--train_setname train_four_phenomena \
--dev_setname dev_four_phenomena \
--test_setname hypernymy_test_seen \
--inferbert_to_load 1
