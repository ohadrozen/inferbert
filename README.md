# InferBert
Datasets and code for the experiments in:   
  
**"Teach the Rules, Provide the Facts: Targeted Relational-knowledge Enhancement for Textual Inference"**, Ohad Rozen, Shmuel Amar, Vered Shwartz and Ido Dagan. *SEM 2021. [link](https://aclanthology.org/2021.starsem-1.8.pdf)

Citation:   
```
@inproceedings{rozen-etal-2021-teach,
    title = "Teach the Rules, Provide the Facts: Targeted Relational-knowledge Enhancement for Textual Inference",
    author = "Rozen, Ohad  and
      Amar, Shmuel  and
      Shwartz, Vered  and
      Dagan, Ido",
    booktitle = "Proceedings of *SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.starsem-1.8",
    doi = "10.18653/v1/2021.starsem-1.8",
    pages = "89--98"
}
```


## Code

**Requirements**: Huggingface pytorch_pretrained_bert library.

**Getting started**: Clone all directories recursively

**Important Note**: Since we combine ramdom weights (S-KAR) with already-trained weights (the original BERT), on small training sets the S-KAR weights might not converge well. To solve that, we repeat the first training phase (See Section 5- Training Prodecedure in the paper) with different random initializations, and evaluate the result on the dev set. We then use the model of the best performing initialization and continue to the secnod training phase. The argument num_of_rand_init controls the number of such initializations. 


### Examples  

#### Example 1
Train InferBert on our training set that combines all 4 phenomena, while adding 10K MNLI examples to the training set, and test it on the hypernymy_unseen_test set. Try 20 different initializations of S-KAR and scan 2 different learning rates:
  
python main.py \
--train_setname train_four_phenomena \
--dev_setname dev_four_phenomena \
--test_setname hypernymy_test_unseen \
--mix_teaching_with_MNLI \
--max_mix_MNLI_examples 10000 \
--learning_rate_vec 2e-5 6e-5 \
--num_of_rand_init 20 \

An output model will be saved at ./source/models/taught_models/[experiment_id], while experiment_id is a running index.


  


#### Example 2
Load the trained model 4 (that was saved at ./source/models/taught_models/4/) and test it on hypernymy_test_seen:
  
python main.py \
--train_setname train_four_phenomena \
--dev_setname dev_four_phenomena \
--test_setname hypernymy_test_seen \
--inferbert_to_load 4

## Datasets
The directory 'inferbert_datasets_combined' includes the training set and the dev set that combines all phenomena as described in our paper. For test, each phenomena has its own test set: seen (including entities that have been seen in the training set) and unseen (new entities that hasn't been seen during training). 'inferbert_datasets_separate' includes the training set of each phenomenon separately.

The location test sets include two types: common- incorporating USA locations with larger population, and rare- with smaller population. The rest results we report in the paper are the average results of the two types. 

**Dataset Fields**
All dataset files are in JSON format. In addition to the 'premise', 'hypothesis' and 'label' fields, each example also consists of the following relevant fields (the rest can be ignored):
* section - For sections '1', '2', or '3' a relation must exist between a tail entity in the premise and a head entity in the hypothesis. For sections '1_other', '2_other' and  '3_other' no relation should exist (though its possible that is exists by chance). For example, for section '2' in a hypernymy dataset, the 'item' exists in the premise and the 'hypernym' in the hypothesis. 
* tail -> head entities: 
Hypernymy datasets: 'item' -> 'hypernym' ('item' indicates the tail entity, i.e. the hyponym, and 'hypernym' indicates the head entity)
Location datasets: 'location' 'country'   (should have been called 'state'. Sorry.)
Country_of_origin datasets: 'company' -> 'country'
Color datasets: 'item' -> 'color'

