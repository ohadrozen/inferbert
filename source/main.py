"""
This code trains and tests our InferBert model as described in our paper "Teach the Rules, Provide the Facts: Targeted Relational-knowledge Enhancement for Textual Inference"
paper is available here: https://aclanthology.org/2021.starsem-1.8.pdf

"""

from __future__ import absolute_import, division, print_function
import sys
#sys.path.append("/home/nlp/ohadr/PycharmProjects/ExtEmb_src/")  # to enable usable of bert model from local folder

import socket
import argparse
import csv
import logging
import os
import random
import datetime
import pickle
import numpy as np
import torch
import utils as ut
from utils import freeze_weights
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from tqdm import tqdm, trange
from modeling_edited import BertConfig, WEIGHTS_NAME, BertWrapper        # BertWrapper wraps BERT to include our S_KAR module.
from modeling_roberta_edited import RobertaConfig, RobertaWrapper
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from tokenization_roberta import RobertaTokenizer

import re
import TemplateProcessor
from TemplateProcessor import InputExample
import gsheets_utils as gu
import copy
import warnings
from country_to_adjective import country2adj
from termcolor import colored
import wordnet_parsing_utils as wnu

BERT_CONFIG_NAME = 'bert_config.json'
NEW_CONFIG_NAME = 'config.json'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To solve an issue with a warning from tokenizer. has no effect on the code itself.

if __name__ == "__main__":
    # parsing here to know whether I should import wordnet_parsing_utils (takes a lot of time due to SpaCy)
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                                                                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--roberta_model", default='roberta-base', type=str, help="RoBERTa model (base or large)")
    parser.add_argument("--task_name", default=None, type=str, help="The name of the task to train.")
    parser.add_argument("--bert_finetuned_model_dir", default="models/BERT_base_84.56", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lm_pretrained_model_dir", default="../BERT_base_pretrained", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--teaching_dir", default=None, type=str, help="The dir where the fine-fine tuning train and test examples are.")
    parser.add_argument("--templates_dir", default=None, type=str, help="The dir where the templates files are, from which the datasets are generated.")
    parser.add_argument("--data_from_templ_dir", default=None, type=str, help="The dir where the auto-generated datasets are.")
    parser.add_argument("--taught_model_dir", default='models/taught_models/', type=str, help="The dir where the Finetuned Taught model is saved. Only relevant for load_taught_model")
    parser.add_argument("--save_model_as", default='', type=str, help="The dir where the Finetuned Taught model is saved. Only relevant for load_taught_model")
    parser.add_argument("--train_group", nargs='+', default=None, type=str, help="the set name of the training set. E.g. 'S1'.")
    parser.add_argument("--test_group", nargs='+', default=None, type=str, help="the set name of the testining set. E.g. 'S2'.")
    parser.add_argument("--train_setname", nargs='+', default=None, type=str, help="the set name of the training set. E.g. 'S1'.")
    parser.add_argument("--test_setname", nargs='+', default=None, type=str, help="the set name of the testining set. E.g. 'S2'.")
    parser.add_argument("--dev_setname", nargs='+', default=None, type=str, help="the set name of the dev set. E.g. 'S2'.")
    parser.add_argument("--EP", default='Combined', type=str, help="the Entailment Phenomenon we'd like to test. E.g. 'Datives'.")
    parser.add_argument("--pretraining_type", type=str, help="The type of pretrained model to load:'MNLI' or 'none'")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--acc_to_compare", type=str, default='S6_cont', help="compare 'MNLI_eval', or 'S6_cont'")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. \nSequences longer than this will be truncated, and sequences shorter \n"
                                                                        "than this will be padded.")

    parser.add_argument("--skip_before", action='store_true', help="Skip steps 1 and 2")
    parser.add_argument("--do_print_log", action='store_true', help="Print log and the end.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_snli_or_mnli", default='mnli', type=str, help="Total batch size for training.")
    parser.add_argument("--do_save_taught_model", action='store_true', help="Whether to save and then load model after fine-fine-tuning.")
    parser.add_argument("--do_load_pretrained_model", action='store_true', help="Whether to load the fine-tuned SNLI/MNLI model finetuned_model.bin.")
    # parser.add_argument("--do_load_taught_model",
    #                     action='store_true',
    #                     help="Whether to load the taught model finetuned_model.bin.")
    parser.add_argument("--mix_teaching_with_MNLI", action='store_true', help="Whether to mix the teaching examples with MNLI examples for the training.")
    parser.add_argument("--comp_before", action='store_true', help="Whether to evaluate before the teaching and compare it to the results after")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", nargs='+', default=[2e-5], type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', nargs='+', type=int, default=[1], help="random seed for initialization")
    parser.add_argument('--no_seed_in_loop', action='store_true', help="If true, initing seed only before the loop and only once. otherwise, initing seed every iter of the loop")
    parser.add_argument('--random_sampling_training_set', action='store_true', help="If false, the training sets will be sampled in the same pseudo-random index all the time for the same set-length")
    parser.add_argument('--semi_random_train', action='store_true', help="If true, initing seed only before the loop and only once. otherwise, initing seed every iter of the loop")
    parser.add_argument('--no_label_balancing', action='store_true', help="Avoids balancing the labels ratio to all have 33% entailment, cont. and neutral in get_templates(). ")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0, help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n" "0 (default value): dynamic loss scaling.\n Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--max_mnli_eval_examples", default=10000, type=int, help="number of examples for eval.")
    parser.add_argument("--train_acc_size", default=10000, type=int, help="Size of train set for accuracy.")
    parser.add_argument("--max_test_examples", default=-1, type=int, help="Maximum number of test examples. if exceeded, we'll only sample this number out of the total number.")
    parser.add_argument("--max_dev_examples", default=-1, type=int, help="Maximum number of dev examples. if exceeded, we'll only sample this number out of the total number.")
    parser.add_argument("--max_train_examples", nargs='+', default=[-1], type=int, help="Maximum number of training examples. if exceeded, we'll only sample this number out of the total number.")
    parser.add_argument("--train_order", default='shuffled', type=str, help="In case of mixing MNLI examples with mine, what's the order of the data: shuffled, MNLI_last or MNLI_first")
    parser.add_argument("--max_mix_MNLI_examples", nargs='+', default=[-1], type=int, help="Maximum number of MNLI examples added to teaching examples.")
    parser.add_argument("--ext_embeddings_type", default='class_auto_attention', type=str, help="The external embedding type. see in modeling_edited.py in BertForSequenceClassification:"
                                                                            "'' - No External Embeddings,"
                                                                            "'fixed_added' - Adding a fixed (manually designed) external embeddings to the word embeddings for each class id"
                                                                            "'class_fixed_added' - Adding the class' word embeddings to the word embeddings"
                                                                            "'first_item_in_class' - Adding the first-item-in-the-class' embeddings to the word embeddings"
                                                                            "'class_auto_added' - Adding regular embeddings by the class ids (using nn.Embeddings())"
                                                                            "'class_auto_concat' - Concatenating regular embeddings by the class ids "
                                                                            "'class_auto_project' - Concatenating regular embeddings by the class ids and then projecting it to original emb size"
                                                                            "'class_fixed_concat' - Concatenating a the class (e.g. 'fruits') word embeddings for each class ids"
                                                                            "'class_fixed_manual_concat' - Concatenating a manually designed fixed embeddings for each class ids"
                                                                            "'first_item_in_class_concat' - Concatenating the first-item-in-the-class' embeddings to the word embeddings")

    parser.add_argument("--concat_embeddings_size", default=12, type=int, help="The size of this external embeddings to be concatenated to BERT's word embeddings")
    parser.add_argument('--fix_position', action='store_true', help="debugs: fixing the position embeddings in BertEmbeddings")
    parser.add_argument('--break_position', action='store_true', help="debugs: switch positions of a few words to dubug the position embeddings")
    parser.add_argument('--freeze_bert', action='store_true', help="freeze all bert weights during training")
    parser.add_argument('--freeze_ext_emb', action='store_true', help="freeze the external embeddings")
    parser.add_argument('--freeze_ext_emb_and_kar', action='store_true', help="freeze the external embeddings")
    parser.add_argument('--short_results', action='store_true', help="display only summary of the results and not all of them")
    parser.add_argument('--debug_tell_label', action='store_true', help="'tells' the model the label of each example using the external embeddings")
    parser.add_argument('--ee_gen_method', default='four_phenomena', type=str, help="for debugging external embeddings:"
                                                                     "'tell label at cls' - 'tells' the model the label of each example by adding the label embedding at cls token"
                                                                     "see more options at tokens_to_ext_emb()")
    parser.add_argument('--ee_gen_method_train', default=None, type=str, help="same as ee_gen_method, but for training only. In all other times (mnli, dev, test) the regular one is the master")
    parser.add_argument('--debug_rule', default=None, type=str, help="backward compatibility for ee_gen_method (replaced by that soon)")
    parser.add_argument('--debug_first_words', action='store_true', help="'tells' the model the label of each example using the first x words: same words 1=2:cont. 1=3:ent. 2=3:neu.")
    parser.add_argument('--test_during_train', action='store_true', help="runs evaluation for the test set after each epoch")
    parser.add_argument("--ext_emb_concat_layer", default=1, type=int, help="what BERT layer to inject the ext embeddings to")
    parser.add_argument('--norm_ext_emb', action='store_true', help="normilizes the ext_embeddings output, like doing for the rest of embeddings")
    # parser.add_argument('--two_steps_train_with_freeze', action='store_true', help="Train ext embeddings while freezing bert and then all the model (see KnowBERT)")
    # parser.add_argument('--force_no_freeze', action='store_true', help="cancels the two_steps training as in args.two_steps_train_with_freeze")
    parser.add_argument("--logging_steps", default=16, type=int, help="average over this number for logging the loss during training")
    parser.add_argument("--ext_learning_rate", nargs='+', default=5e-3, type=float, help="Learning rate for external embeddings weights")
    parser.add_argument("--kar_h_proj_size", default=10, type=int, help="The size of this external embeddings to be concatenated to BERT's word embeddings")
    parser.add_argument("--kar_ext_emb_size", default=768, type=int, help="The size of external embeddings when using KAR (KnowBert)")
    parser.add_argument('--config_ext_emb_method', default=None, type=str, help="internal use only! Don't input from command line! will be overwriten by add_to_config()")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="number of BERT layers")
    parser.add_argument('--freeze_input', nargs='+', type=str, default=[''], help="input for the interactive freeze")
    parser.add_argument('--print_trained_layers', action='store_true', help="display what layer/module was trained")
    parser.add_argument("--temp_models_dir", default=None, type=str, help="The dir where the temp models are saved.")
    parser.add_argument("--num_of_rand_init", default=30, type=int, help="Number of initialization of model and training external embeddings (taking the best) before moving to training the rest of model")
    parser.add_argument("--pretrained_model_to_load", default='bert_mnli', type=str, help="can be: 'bert_lm', 'bert_mnli', model_filnename (e.g. '../taught_model/tmp1.bin') or None(--> random init)")
    parser.add_argument("--test_before", action='store_true', help="Evaluate Test set and MNLI set right after loading the model, before any finetuning")
    parser.add_argument("--ngram_classifier", type=str, default='svm', help="Evaluate Test set and MNLI set right after loading the model, before any finetuning")
    parser.add_argument("--is_sparse", action='store_true', help="Evaluate Test set and MNLI set right after loading the model, before any finetuning")
    parser.add_argument("--num_train_epochs_ext", type=int, default=3, help="number of training epochs for external weights training only (freeze_type='all_but_ext_emb')")
    parser.add_argument("--cached_features_dir", default=None, type=str, help="Directory for saving the cached features dumps.")
    parser.add_argument("--overwrite_cached_features", action='store_true', help="Overwrite the cached features file")
    parser.add_argument("--uniform_ext_emb", action='store_true', help="activate self.ext_embeddings.weight.data.uniform_(-12.49, 4.69) in modeling_edited.py")
    parser.add_argument("--notes", type=str, default="", help="Notes to add to wandb")
    parser.add_argument("--no_kar_norm", action='store_true', help="if True, not normalizing inside BertKAROutput in modelind_edited.py")
    parser.add_argument("--save_every_init", action='store_true', help="for ext_emb save a model for each init")
    parser.add_argument("--save_every_grid_step", action='store_true', help="for grid search save a model for each hyperparam")
    parser.add_argument("--mnli_examples_start_ind", type=int, default=0, help="starting index for taking mnli examples")
    parser.add_argument("--duplicate_train_examples", type=int, default=1, help="number of duplications of the same training examples in the training set, not including MNLI examples")
    parser.add_argument("--weight_to_challenge_set", nargs='+', type=int, default=4, help="if !=1 gives a different wieght to the challenge set examples (if they have 'info' attribute and info['example_type']=='challenge_set'."
                                                                                "This weight is given by multiplying the weight in the loss result")
    parser.add_argument("--model_debug_type", default=None, type=str, help="controlling different configurations of the model inside modeling_edited.py")
    parser.add_argument("--debug_second_freeze_step", action='store_true', help="makes 'bert' step in training freeze only ext_emb instead of ext_emb+KAR")
    parser.add_argument("--model_type",  default='bert', type=str, help="'BERT' or 'RoBERTa'")
    parser.add_argument("--ext_learning_rate_vec", nargs='+', default=[6e-3], type=float, help="list of learning rate for the grid search")
    parser.add_argument("--learning_rate_vec", nargs='+', default=5e-3, type=float, help="list of learning rate for the grid search")
    parser.add_argument("--wandb_off", action='store_true', help="if True, doesn't run wandb")
    parser.add_argument("--load_ext_emb_model", default=None, type=str, help="skipping ext_emb finetuning and loading this model")
    parser.add_argument("--inferbert_to_load", default=None, type=str, help="finetuned Inferbert model to load (e.g. '../taught_models/2789_hypernymy.bin')")
    parser.add_argument("--hypernymy_mode", default='no_chunk', type=str, help="how to extract candidates from sentence for hypernymy")
    parser.add_argument("--finetune_vanilla", action='store_true', help="if True, finetune Bert vanivlla, without Ext Emb")
    parser.add_argument("--nb_vanilla_epochs", default=1, type=float, help="Total number of training epochs to perform for Vanilla only.")

    args = parser.parse_args()
    excluded_loop_args = ['ext_learning_rate_vec', 'learning_rate_vec', 'freeze_input', 'train_setname', 'dev_setname', 'test_setname', 'train_group', 'dev_group', 'test_group']     # arguments that can be lists, but we don't want to loop over this list
    args_copy = copy.deepcopy(args)


    if not args.finetune_vanilla:
        if args.EP in ['Dir_Hypernymy', 'Location', 'Trademark']:
            local_wiki = wnu.load_local_wiki()
        elif args.EP in ['Color']:
            local_wiki_features = wnu.load_local_wiki_features()
        elif args.EP in ['Combined']:
            local_wiki_features = wnu.load_local_wiki_features()
            local_wiki = wnu.load_local_wiki()

    FINE_TUNED_NAME = 'finetuned_model.bin'

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    if axis==0:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x,1).reshape(-1,1))
        return e_x / np.sum(e_x,1).reshape(-1,1)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, ext_emb_ids, segment_ids, label_id, example_weight, example_source):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.ext_emb_ids = ext_emb_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_weight = example_weight
        self.example_source = example_source


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _read_txt(cls, input_file, quotechar=None):
        """Reads 3 lines value file of the format:
        line0: premis
        line1: hypothesis
        line2: label
        line3: blank line
        ...
        """
        with open(input_file, "r") as f:
            content = f.readlines()
            lines = []
            items = []
            template_id = ''
            ind = 0
            for i, line in enumerate(content):
                str = '' if line==[] else line.lower().strip()
                if ind % 4 == 0 and (str == '' or str[0]=='#'):
                    t_loc = str.find('template:')           # if there is a template mark at the examples file: "### Template:2 ##### ==> template_id=2
                    if t_loc > 0:
                        template_id = str[t_loc+9:t_loc+11].strip()
                    continue
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                if ind%4 == 3:
                    assert (str==''), 'every 4th line must be empty. In line %d at %s' % (i, input_file)
                    items.append(template_id)       # adding template_id as the last item
                    lines.append(items)
                    items = []
                else:
                    items.append(str)

                ind += 1
            if ind%4 == 3:
                # just in case file ended without enpty line at the end
                lines.append(items)
            assert ind % 4 in [0, 3]
            return lines

    def _read_templates(cls, template_file):
        """Reads 3 lines value file of the format:
        line0: premis
        line1: hypothesis
        line2: label
        line3..: templates
                blank line
        ...
        """
        def parse_expression(s):
            if len(list(re.finditer(':',s)))==2:      # linear numbers list: 4:2:10 ==> 4; 6; 8; 10
                args = s.split(':')
                insts = [str(i) for i in range(int(args[0]), int(args[2])+1, int(args[1])  )]
            elif len(list(re.finditer(':',s)))==1:      # linear numbers list: 4:10 ==> 4;5;6;7;8;9;10
                args = s.split(':')
                insts = [str(i) for i in range(int(args[0]), int(args[1])+1)]
            return insts



        def parse_tepmlate_str(s):
            loc = s.find('*:', 2)
            template = s[:loc+1]
            instances = s[loc+2:].lower().split(';')
            instances_t = [s.strip() for s in instances if s != '']   # remove empty instances
            instances_n = []
            for inst in instances_t:  # looking for and parsing instances expressions (e.g. <<20:2:30>> ==> 20,22,24,...
                start_loc = inst.find('<<')
                if start_loc >= 0:
                    end_loc = inst.find('>>')
                    assert end_loc>0, 'Error in _read_template lind %d, file %s: after "<<" must come ">>"' % (i, template_file)
                    expr_list = parse_expression(inst[start_loc+2:end_loc])
                    instances_n.extend(expr_list)
                else:
                    instances_n.append(inst)
            assert s[0] == '*', 'Error in _read_template lind %d, file %s: After Label line the Template must come' % (i, template_file)
            assert loc > -1, 'Error in _read_template lind %d, file %s: template must be of the form *TEMPLATE*:  ... ; ... ;' % (i, template_file)
            return template, instances_n

        with open(template_file, "r", encoding="utf8", errors='ignore') as f:
            examples = []
            ind = 0
            template_found = 0
            content = f.readlines()
            for i, line in enumerate(content):
                s = '' if line == [] else line.lower().strip()

                if s!='' and s[0]=='#': continue
                if ind % 5 == 0 and s=='': continue

                if ind % 5 == 0:                # first line: Premise
                    p = s
                    templates = []
                if ind % 5 == 1: h = s        # second line: Hypothesis
                if ind % 5 == 2: l = s        # third line: label
                if ind % 5 == 3:                # Froth line and more: templates
                    if s == '':   # in case no template reported
                        ind +=1
                    else:           # template reported
                        templates.append(parse_tepmlate_str(s))
                        template_found = 1
                        ind -=1                     # keep looking for more templates lines
                if ind % 5 == 4:                # end of example. append sentences and templates
                    examples.append((p, h, l, templates))
                    assert not ((template_found==0) and ((p.find('*')>=0) or (h.find('*')>=0))), 'Error in _read_template: "*" found but no template found. line %d, file %s' % (i, template_file)
                    template_found = 0
                ind += 1
                assert not (ind%5 == 4 and s!=''), 'Error in _read_template: line should be empty. line %d, file %s' % (i, template_file)
            assert ind % 5 in [3, 4], 'Error in _read_template: finished reading the file in the middle of an example. line %d, file %s' % (i, template_file)
            if ind % 5 == 3:                        # when there are no templates, the process needs to add the last example to the list
                examples.append((p, h, l, templates))
            return examples

    def _dump_templates(cls, examples, output_file):
        # generates a file with all possible examples from templates
        print("generating %s from template..."%output_file)
        with open(output_file, "w") as f:
            for t_id, (p, h, l, templates) in enumerate(examples):
                f.write('###### Template:%d ########################################################################\n' % t_id)
                if templates==[]:  # no template
                    f.write(p + '\n')
                    f.write(h + '\n')
                    f.write(l + '\n')
                    f.write('\n')
                else:               # template found
                    f.write('# '+ p + '\n')
                    f.write('# '+ h + '\n')
                    f.write('# '+ l + '\n')
                    for template, instances in templates:
                        f.write('# '+ template + ':' + str(instances) + '\n')
                    f.write('\n')
                    ind_vec = [0]*len(templates)
                    while True:                                                     # scan all possible combinations of instances
                        p_n = p
                        h_n = h
                        for t, (template, instances) in enumerate(templates):       #instanciate all instances instead of the template variables
                            p_n_before = p_n; h_n_before = h_n
                            inst = instances[ind_vec[t]].strip()
                            p_n = p_n.replace(template, inst)
                            h_n = h_n.replace(template, inst)
                            assert not (p_n_before==p_n and h_n_before==h_n), 'for template "%s" there is no instance in premise="%s" and hypothesis="%s" at file %s'%(template, p, h, output_file)
                        assert p_n.find('*')<0, 'the premise was not fully instanced: %s'%(p_n)
                        assert h_n.find('*')<0, 'the hypothesis was not fully instanced: %s'%( h_n)
                        f.write(p_n + '\n')
                        f.write(h_n + '\n')
                        f.write(l + '\n')
                        f.write('\n')
                        ind_vec[-1] += 1
                        for t in range(len(ind_vec)-1, 0, -1):
                            if ind_vec[t] == len(templates[t][1]):
                                ind_vec[t] = 0
                                ind_vec[t-1] += 1
                        if ind_vec[0] == len(templates[0][1]): break
            print("finished writing.")


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, max_examples=1000000):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_snli_dev_examples(self, data_dir = '../glue_data/SNLI/parsed_sentences'):
        """See base class."""
        return self._create_snli_examples(data_dir, 'dev')

    def get_teaching_train_templates(self,data_dir):
        templates = self._read_templates(os.path.join(data_dir, "train_templates.txt"))
        self._dump_templates(templates, os.path.join(data_dir, "train.txt"))
        print('reading train.txt')
        return self._create_teaching_examples(
            self._read_txt(os.path.join(data_dir, "train.txt"))), templates

    def get_teaching_test_templates(self,data_dir):
        templates = self._read_templates(os.path.join(data_dir, "test_templates.txt"))
        self._dump_templates(templates, os.path.join(data_dir, "test.txt"))
        print('reading test.txt')
        return self._create_teaching_examples(
            self._read_txt(os.path.join(data_dir, "test.txt"))), templates

    def get_teaching_train_examples(self,data_dir):
        return self._create_teaching_examples(
            self._read_txt(os.path.join(data_dir, "train.txt")))

    def get_teaching_test_examples(self,data_dir):
        return self._create_teaching_examples(
            self._read_txt(os.path.join(data_dir, "test.txt")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_finetune_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_teaching_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            template_id = line[3]
            guid = 'Template:%s' % template_id
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def setup_logger(formatter, name, log_file, level=logging.INFO, ):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def my_logger(message, debug_only=0, highlight=None, color=None):
    # logging the message. if debug_only=1 it only logs it to logger_debug (the detailed logger)
    def maybe_highlight(message):
        if highlight is not None:  # adding '@@@@@@@@@@...' before
            highlight_str = '\n%s ' % (highlight[0] * highlight[1])
            message = highlight_str + message + highlight_str
            print(colored(highlight[0] * highlight[1], color))
        return message

    global logger_debug, logger

    message = maybe_highlight(message)
    logger_debug.info(message)
    if debug_only == 0:
        logger.info(message)
        print(colored(message, color))

def convert_examples_to_features(args, examples, max_length, tokenizer, ext_embeddings_type='', break_position=False, data_type=''):
    if args.model_type == 'bert':
        return convert_examples_to_features_bert(args, examples, max_length, tokenizer, ext_embeddings_type=ext_embeddings_type, break_position=break_position, data_type=data_type)
    elif args.model_type == 'roberta':
        return convert_examples_to_features_roberta(args, examples, max_length, tokenizer, ext_embeddings_type=ext_embeddings_type, break_position=break_position, data_type=data_type)


def get_example_weight(example):
    example_weight = args.weight_to_challenge_set if (hasattr(example, 'info') and example.info and example.info['example_type']=='challenge_set') else 1
    example_source = 1 if (hasattr(example, 'info') and example.info and example.info['example_type'] == 'challenge_set') else 0      # give 1 to examples from challenge_set
    return example_weight, example_source



def convert_examples_to_features_bert(args, examples, max_seq_length, tokenizer, ext_embeddings_type='', break_position=False, data_type=''):
    """Loads a data file into a list of `InputBatch`s."""
    
    use_ext_embeddings = (ext_embeddings_type != '')
    label_map = {'contradiction': 0, 'entailment':1, 'neutral': 2}

    features = []
    ee_gen_method = args.ee_gen_method_train if (data_type == 'train' and args.ee_gen_method_train is not None) else args.ee_gen_method

    my_logger("Converting examples to features (data_type = %s), using ee_gen_method=%s:"%(data_type, ee_gen_method))
    # print_every = 100 if args.ee_gen_method in ['hypernym_pairs_wordnet'] else 1000
    counters = dict(debug_hyper_counter=0, debug_nonhyper_counter=0, debug_missed_hyper_counter=0)

    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting to features")):
        # if ex_index%print_every==0 and ex_index>print_every-1:
        #     my_logger("%d/%d examples converted"%(ex_index, len(examples)))

        label_id = label_map[example.label]
        example_weight, example_source = get_example_weight(example)
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        # for debugging:
        def switch_pos(tokens, i, j):
            token_tmp = tokens[i]
            tokens[i] = tokens[j]
            tokens[j] = token_tmp
        # for debugging:
        if break_position:
            switch_pos(tokens, 0, 5)
            switch_pos(tokens, 1, 4)
            switch_pos(tokens, 2, 3)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        seg_b_loc = segment_ids.index(1)
        ext_emb_ids = tokens_to_ext_emb(tokens, input_ids, seg_b_loc, use_ext_embeddings, tokenizer, label_id, ee_gen_method, example, args.model_type, tokens_a=tokens_a, tokens_b=tokens_b)


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)


        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        ext_emb_ids += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(ext_emb_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length


        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              ext_emb_ids=ext_emb_ids,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              example_weight=example_weight,
                              example_source=example_source,
                              ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_by_label(out, labels, source):
    outputs = np.argmax(out, axis=1)
    good_by_label = np.array([np.sum((outputs == labels) & (labels == 0)),
                     np.sum((outputs == labels) & (labels == 1)),
                     np.sum((outputs == labels) & (labels == 2))])

    good_by_source = np.array([np.sum((outputs == labels) & (source == 0)), np.sum((outputs == labels) & (source == 1))])

    dist_by_label = np.array([
        np.sum((outputs == 0) & (labels == 0)),
        np.sum((outputs == 1) & (labels == 0)),
        np.sum((outputs == 2) & (labels == 0)),
        np.sum((outputs == 0) & (labels == 1)),
        np.sum((outputs == 1) & (labels == 1)),
        np.sum((outputs == 2) & (labels == 1)),
        np.sum((outputs == 0) & (labels == 2)),
        np.sum((outputs == 1) & (labels == 2)),
        np.sum((outputs == 2) & (labels == 2)),
    ])
    return good_by_label, dist_by_label, good_by_source


def pred_distribution(out):
    outputs = np.argmax(out, axis=1)
    return np.array([np.sum(outputs == 0),np.sum(outputs == 1),np.sum(outputs == 2)])


def print_log(log, last_lines = 0):
    for str in log[-last_lines:]:
        my_logger(str)

def dump_log(filename, log, append = False):
    app_str = 'a+' if append else 'w'
    my_logger("Dumping log to %s"%filename, 1)
    with open(filename, app_str) as f:
        for line in log:
            f.write(str(line) + '\n')

def dump_df(filename, csv_str):
    my_logger("Dumping DataFrame to %s..."%filename, 1)
    with open(filename, 'w') as f:
        f.write(csv_str)


def template2str(templates, ind = -1):
    """ converts templates list to a string list according to the index
    Input:
        Templates list
        Ind - index of the wanted template. If ind=-1, return all templates

    Output: list of strings representing the template
    """
    str_l = []
    templates_to_extract = templates if ind==-1 else templates[ind]
    for t_id, (p, h, l, arguments, _) in enumerate(templates_to_extract):
        str_l.append('###### Template:%d ########################################################################\n' % t_id)
        if arguments == []:  # no template
            str_l.append(p)
            str_l.append(h)
            str_l.append(l)
            str_l.append('\n')
        else:  # template found
            for h_ind, (hi, li) in enumerate(zip(h, l)):  # looping over different Hypothesises
                if len(h) > 1:
                    str_l.append('### Hypothesis %d:\n' % h_ind)
                str_l.append('# ' + p )
                str_l.append('# ' + hi)
                str_l.append('# ' + li)
                for arg, instances in arguments:
                    str_l.append('# ' + arg + ': ' + TemplateProcessor.inst2str(instances) )
                str_l.append('\n')
    return str_l


def get_loaded_model_trained_sets(model_dir):
    info_filename = os.path.join(model_dir, 'training_info.txt')
    with open(info_filename , 'r') as handle_info:
        content = handle_info.readlines()
        loaded_model_trained_sets = re.search(r'(?<=Trained on:).*', content[0], re.IGNORECASE).group(0).strip()
    return loaded_model_trained_sets

def get_subset_results(examples, config, before_or_after='after'):
    """
    :param examples: train_examples or test_examples. a list of examples
    :param config: a list of attributes to look for in the examples list. E.g. config = {'hypothesis_id': 0, 'set.name': 'S6-MNLI'}
    :return:  acc, total_relevant, total_good
    """
    total_relevant = total_good = 0
    for e in examples:
        relevant_example = True
        for key_s in config.keys():
            keys = key_s.split('.')
            if len(keys) == 2:
                key1, key2 = keys
                if not e.info[key1][key2]==config[key_s]:
                    relevant_example = False
            elif len(keys) ==1:
                key1 = keys[0]
                if not e.info[key1]==config[key_s]:
                    relevant_example = False
            else:
                raise ValueError ("only one dot is allowed in the config key string")
        if relevant_example:
            total_relevant +=1
            total_good += e.good_before if before_or_after=='before' else e.good_after
    acc = total_good/total_relevant
    return acc, total_relevant, total_good

def ind2sentence(indices):
    with open('bert-base-uncased-vocab_edited.txt','r') as f:
        lines = f.readlines()
        s = ''
        for ind in indices:
            s += lines[ind].strip() + ' '
    return s

def word_piece_connected_bert(tokens, input_ids):
    # Input: tokens of wordpiece tokenizer output and their id's
    # Output: the connected token list + new2old_ind converter
    # For example (BERT):
    # word_to_find = 'broccoli'
    # tokens =  "[CLS] '  mom  ##will send you gifts   bro  ##cco ##li  and  will ##send  you  bro ##cco ##li fruits for  the  return journey , ' shouted the quartermaster . [SEP] ' mom will send you gifts and bro ##cco ##li for the return journey , ' shouted the quartermaster . [SEP] ".split()
    # input_ids =[101, 18, 202, 313,   414, 515, 616,   717,  838,  949, 111, 222,  333,   444, 555, 666,  18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, ]
    # old_ind =  [0,   1,   2,   3,    4,   5,   6,      7,   8,    9,   10,  11,   12,    13,  14,  15,   16,  17     18  ]
    # new_ind=   [0,   1,   2,         3,   4,   5,      6,               7,   8,           9,  10,             11,    12,    13]

    old_ind = list(range(len(input_ids)))
    new_ind = []
    new2old_ind = []
    s_new = []
    for i in old_ind:
        if tokens[i][:2] != '##':
            new_ind.append(input_ids[i])
            new2old_ind.append(i)
            s_new.append(tokens[i])
        else:
            wt = s_new.pop()
            wt += tokens[i][2:]
            s_new.append(wt)
    return s_new, new2old_ind


def word_piece_connected_roberta(tokens, input_ids):
    # Input: tokens of wordpiece tokenizer output and their id's
    # Output: the connected token list + new2old_ind converter
    # For example (BERT):
    # word_to_find = 'broccoli'
    # tokens =  # ['<s>', 'The', 'Ġroute', 'Ġcoming', ... 'Ġeasy', '.', '</s>', '</s>', 'The', 'Ġroute', 'Ġcoming'... '</s>', '<pad>', ...]
    # tokens(bert) =  "[CLS] '  mom  ##will send you gifts   bro  ##cco ##li  and  will ##send  you  bro ##cco ##li fruits for  the  return journey , ' shouted the quartermaster . [SEP] ' mom will send you gifts and bro ##cco ##li for the return journey , ' shouted the quartermaster . [SEP] ".split()
    # input_ids =[101, 18, 202, 313,   414, 515, 616,   717,  838,  949, 111, 222,  333,   444, 555, 666,  18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, ]
    # old_ind =  [0,   1,   2,   3,    4,   5,   6,      7,   8,    9,   10,  11,   12,    13,  14,  15,   16,  17     18  ]
    # new_ind=   [0,   1,   2,         3,   4,   5,      6,               7,   8,           9,  10,             11,    12,    13]

    ##TODO: I think that there is a bug with location of the period at end of sentence ==> 'LastWordToken.' instead of 'LastWordToken'
    old_ind = list(range(len(input_ids)))
    new_ind = []
    new2old_ind = []
    s_new = []
    for i in old_ind:
        if i == 0 or \
                tokens[i][0] == 'Ġ' \
                or tokens[i] in ['<s>', '</s>'] \
                or (i > 0 and tokens[i-1] in ['<s>', '</s>']):        # a new word condition
            new_ind.append(input_ids[i])
            new2old_ind.append(i)
            if tokens[i][0] == 'Ġ':         # if token starts with 'Ġ' - remove it when creating s_new
                s_new.append(tokens[i][1:])
            else:
                s_new.append(tokens[i])
        else:                           # additional piece of the current word
            wt = s_new.pop()
            wt += tokens[i]
            s_new.append(wt)
    return s_new, new2old_ind

def debug_word_piece(tokens, s_new, new2old_ind):
    print('\t'.join(["%d %s %3s" % (i, c, '') for i, c in enumerate(tokens)]))
    print('\t'.join(["%d %s %3s" % (new2old_ind[i], c, '') for i, c in enumerate(s_new)]))


def word_piece_connected(tokens, input_ids, model_type):
    # Input: tokens of wordpiece tokenizer output and their id's
    # Output: the connected token list + new2old_ind converter
    # For example:
    # word_to_find = 'broccoli'
    # tokens =  "[CLS] '  mom  ##will send you gifts   bro  ##cco ##li  and  will ##send  you  bro ##cco ##li fruits for  the  return journey , ' shouted the quartermaster . [SEP] ' mom will send you gifts and bro ##cco ##li for the return journey , ' shouted the quartermaster . [SEP] ".split()
    # input_ids =[101, 18, 202, 313,   414, 515, 616,   717,  838,  949, 111, 222,  333,   444, 555, 666,  18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, ]
    # old_ind =  [0,   1,   2,   3,    4,   5,   6,      7,   8,    9,   10,  11,   12,    13,  14,  15,   16,  17     18  ]
    # new_ind=   [0,   1,   2,         3,   4,   5,      6,               7,   8,           9,  10,             11,    12,    13]

    if model_type == 'bert':
        return word_piece_connected_bert(tokens, input_ids)
    if model_type == 'roberta':
        return word_piece_connected_roberta(tokens, input_ids)
    raise ValueError("Wrong model_type")



def tokens_to_ext_emb(tokens, input_ids, seg_b_loc, use_ext_embeddings, tokenizer, label_id=None, ee_gen_method=None, example=None, model_type=None, tokens_a='', tokens_b=''):
    """
    Implementation of the relation extractor. See Section 4.2 in paper, as well as Figure 1.
    We place the relevant id of the relation embeddings (e.g. hypernymy_head=1) in the index of the first token of the relevant word (e.g. in case the hyponym consists of a few tokens)

    # Returns a list of zeros, except of locations of tokens that are head of the relation or tail.
    # For example, for the input: "[CLS] I see apples [SEP] I see fruits" the output would be [0, 0, 0, 1, 0, 0, 0, 2] since
    # the id of the hypernymy_head is 1 and hypernymy_tail is 2.
    #   Input: tokens of wordpiece tokenizer output and their id's + the (complete) word of which we want to find the index
    #   Output: the index of the first piece in the word piece.

    # Example for finding the indices of the relevant tokens:
    # word_to_find = 'broccoli'
    # tokens =  "[CLS] '  mom  ##will send you gifts   bro  ##cco ##li  and  will ##send  you  bro ##cco ##li fruits for  the  return journey , ' shouted the quartermaster . [SEP] ' mom will send you gifts and bro ##cco ##li for the return journey , ' shouted the quartermaster . [SEP] ".split()
    # input_ids =[101, 18, 202, 313,   414, 515, 616,   717,  838,  949, 111, 222,  333,   444, 555, 666,  18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, ]
    # old_ind =  [0,   1,   2,   3,    4,   5,   6,      7,   8,    9,   10,  11,   12,    13,  14,  15,   16,  17     18  ]
    # new_ind=  [0,   1,   2,         3,   4,   5,      6,               7,   8,           9,  10,             11,    12,    13]
    # Result: [7, 14]
    """



    def get_offset_new(tokens_a_new, model_type):
        # offset from location in tokens_a_new to location here, so we can activate new2old_ind[] on it.
        if model_type == 'bert':
            # ['[CLS]'] + tokens_a_new + ['[SEP]']+ tokens_b_new+ ['[SEP]']
            return len(tokens_a_new) + 2
        if model_type == 'roberta':
            # ['<s>'] + tokens_a_new + ['</s>', '</s>'] + tokens_b_new+ ['</s>']
            return len(tokens_a_new) + 3

    def validate_new_ids(word, input_ids, word_ind, new2old_ind, offset):
        retrieved_token = tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[word_ind + offset]]])[0]
        relevant_word_piece = word[:len(retrieved_token)]
        if retrieved_token != relevant_word_piece:
            my_logger(f"index of wordpiece doesn't match the index of original sentence: '{retrieved_token}' != '{relevant_word_piece}'")
            my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
            print('********************************************************')

    def validate_new_ids_for_country(country, country_adj, input_ids, word_ind, new2old_ind, offset):
        retrieved_token = tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[word_ind + offset]]])[0]
        country_cond = retrieved_token == country[:len(retrieved_token)]
        country_adj_cond = retrieved_token == country_adj[:len(retrieved_token)]
        if not (country_cond or country_adj_cond):
            my_logger(f"\nindex of wordpiece doesn't match the index of original sentence: '{retrieved_token}' != '{country[:len(retrieved_token)]}' and != '{country_adj[:len(retrieved_token)]}'")
            my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
            print('********************************************************')

    def country2adj_func(country):
        country = country[0].upper() + country[1:]  # capitalizing country string
        if country in country2adj:
            return country2adj[country]
        else:
            return country

    global item2class, class2ind
    ext_emb = [0] * len(input_ids)

    # returning 0 vector when not using any knowledge enhancement method
    if use_ext_embeddings == False:
        return ext_emb

    if ee_gen_method == 'four_phenomena':    # adding ext_id=1 to any hypernym and =2 to its hyponym, using wordnet - but exclude nouns that appear in both sentences
        s_new, new2old_ind = word_piece_connected(tokens, input_ids, model_type)  # s_new is the connected tokens list.
        tokens_a_new, tokens_b_new = word_piece_connected(tokens_a, [1]*len(tokens_a), model_type)[0], word_piece_connected(tokens_b, [1]*len(tokens_b), model_type)[0]   # connecting the word_pieces together
        offset_new = get_offset_new(tokens_a_new, model_type)  # offset from location in tokens_a_new to location here: ['[CLS]']+ tokens_a_new + ['[SEP]']+ tokens_b_new+ ['[SEP]'] , so we can activate new2old_ind[] on it.
        relation2id_vocab = dict(location_head=1, location_tail=2, color_head=3, color_tail=4, trademarks_head=5, trademarks_tail=6, hypernymy_head=7, hypernymy_tail=8)

         ## Location
        pairs = wnu.find_location_country_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, local_wiki, is_print=False, filter_repeat_word=True)
        # {'alabama': {'location': ['trussville'],
        #              'location_ind': [1],
        #              'country_ind': [0]}}
        offset = (1, offset_new)
        for country in pairs:     # hypo in premise -> country in hypothesis
            for location, location_i in zip(pairs[country]['location'], pairs[country]['location_ind']):
                ext_emb[new2old_ind[location_i + offset[0]]] = relation2id_vocab['location_tail']
                if not (tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0] == location[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0])]):
                    warnings.warn(f"index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0]}' != '{location[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0])]}'")
                    my_logger(f"Warning! index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0]}' != '{location[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0])]}'")
                    my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
                    print('********************************************************')
            for country_i in pairs[country]['country_ind']:
                ext_emb[new2old_ind[country_i + offset[1]]] = relation2id_vocab['location_head']
                if not (tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0] == country[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0])]):
                    my_logger(f"index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0]}' != '{country[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0])]}'")
                    my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
                    print('********************************************************')

        ## Color
        pairs = wnu.find_color_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, local_wiki_features, is_print=False, filter_repeat_word=True)
        # pairs = {'grey': {'noun': ['steel'],
        #           'noun_ind': [6],
        #           'feature_ind': [9],
        #           'type': 'color'}}
        offset = (1, offset_new)
        for color in pairs:     # hypo in premise -> country in hypothesis
            for noun, noun_i in zip(pairs[color]['noun'], pairs[color]['noun_ind']):
                ext_emb[new2old_ind[noun_i + offset[0]]] = relation2id_vocab['color_tail']
                validate_new_ids(noun, input_ids, noun_i, new2old_ind, offset[0])
            for feature_i in pairs[color]['feature_ind']:
                ext_emb[new2old_ind[feature_i + offset[1]]] = relation2id_vocab['color_head']
                validate_new_ids(color, input_ids, feature_i, new2old_ind, offset[1])

        ## Trademarks
        pairs = wnu.find_trademark_country_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, local_wiki, is_print=False, filter_repeat_word=True)
        # pairs = {'grey': {'noun': ['steel'],
        #           'noun_ind': [6],
        #           'feature_ind': [9],
        #           'type': 'company'}}
        offset = (1, offset_new)
        for country in pairs:     # hypo in premise -> country in hypothesis
            for noun, noun_i in zip(pairs[country]['company'], pairs[country]['company_ind']):
                ext_emb[new2old_ind[noun_i + offset[0]]] = relation2id_vocab['trademarks_tail']
                validate_new_ids(noun, input_ids, noun_i, new2old_ind, offset[0])
            for feature_i in pairs[country]['country_ind']:
                ext_emb[new2old_ind[feature_i + offset[1]]] = relation2id_vocab['trademarks_head']
                validate_new_ids_for_country(country, country2adj_func(country).lower(), input_ids, feature_i, new2old_ind, offset[1])

        ## Hypernymy
        pairs, _ = wnu.find_hypernymy_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, filter_repeat_word=True, mode=args.hypernymy_mode)
        offset = (1, offset_new)
        for hyper in pairs:     # hypo in premise -> hyper in hypothesis
            for hypo, hypo_i in zip(pairs[hyper]['hypo'], pairs[hyper]['hypo_ind']):
                ext_emb[new2old_ind[hypo_i + offset[0]]] = relation2id_vocab['hypernymy_tail']
                validate_new_ids(hypo, input_ids, hypo_i, new2old_ind, offset[0])
            for hyper_i in pairs[hyper]['hyper_ind']:
                ext_emb[new2old_ind[hyper_i + offset[1]]] = relation2id_vocab['hypernymy_head']
                validate_new_ids(hyper, input_ids, hyper_i, new2old_ind, offset[1])
    elif ee_gen_method == 'tell_label_at_cls':  # WORKS. cheating and 'telling' the model the lable, to see if it reaches 100% accuracy using ext. embeddings
        loc = 1
        ext_emb[loc] = label_id
    elif ee_gen_method == 'all_zeros':  #
        pass
    elif ee_gen_method == 'hypernym_old_pairs_wordnet_filtered_clean':  # extracting only pairs with Relation=Hypernymy. Adding ext_id=1 to any hypernym and =2 to its hyponym, using wordnet - but exclude nouns that appear in both sentences
        # ext_emb = list(np.random.randint(3, 999, size=len(ext_emb)))
        s_new, new2old_ind = word_piece_connected(tokens, input_ids, model_type)  # s_new is the connected tokens list.
        tokens_a_new, tokens_b_new = word_piece_connected(tokens_a, [1]*len(tokens_a), model_type)[0], word_piece_connected(tokens_b, [1]*len(tokens_b), model_type)[0]   # connecting the word_pieces together
        offset_new = get_offset_new(tokens_a_new, model_type)  # offset from location in tokens_a_new to location here: ['[CLS]']+ tokens_a_new + ['[SEP]']+ tokens_b_new+ ['[SEP]'] , so we can activate new2old_ind[] on it.
        (p_h_pairs, h_p_pairs), (doc1, doc2) = wnu.find_hyper_hypo_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, filter_repeat_word=True)
        for pairs, offset in zip([p_h_pairs, h_p_pairs], [(1, offset_new), (offset_new, 1)]):
            for hyper in pairs:     # hypo in premise -> hyper in hypothesis
                for hypo, hypo_i in zip(pairs[hyper]['hypo'], pairs[hyper]['hypo_ind']):
                    ext_emb[new2old_ind[hypo_i + offset[0]]] = 2
                    if not (wnu.to_singular(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hypo_i + offset[0]]]])[0]) == wnu.to_singular(hypo[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hypo_i + offset[0]]]])[0])])):
                        warnings.warn(f"index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hypo_i + offset[0]]]])[0]}' != '{hypo[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hypo_i + offset[0]]]])[0])]}'")
                        my_logger(f"Warning! index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hypo_i + offset[0]]]])[0]}' != '{hypo[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hypo_i + offset[0]]]])[0])]}'")
                        my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
                        (p_h_pairs_, h_p_pairs_), (doc1_, doc2_) = wnu.find_hyper_hypo_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new)
                        print('********************************************************')

                for hyper_i in pairs[hyper]['hyper_ind']:
                    ext_emb[new2old_ind[hyper_i + offset[1]]] = 1
                    if not (wnu.to_singular(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hyper_i + offset[1]]]])[0]) == wnu.to_singular(hyper[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hyper_i + offset[1]]]])[0])])):
                        my_logger(f"index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hyper_i + offset[1]]]])[0]}' != '{hyper[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[hyper_i + offset[1]]]])[0])]}'")
                        my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
                        (p_h_pairs_, h_p_pairs_), (doc1_, doc2_) = wnu.find_hyper_hypo_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new)
                        print('********************************************************')
    elif ee_gen_method == 'location_wiki_filtered_clean':  # extracting only pairs with Relation=Location.
        s_new, new2old_ind = word_piece_connected(tokens, input_ids, model_type)  # s_new is the connected tokens list.
        tokens_a_new, tokens_b_new = word_piece_connected(tokens_a, [1]*len(tokens_a), model_type)[0], word_piece_connected(tokens_b, [1]*len(tokens_b), model_type)[0]   # connecting the word_pieces together
        offset_new = get_offset_new(tokens_a_new, model_type)  # offset from location in tokens_a_new to location here: ['[CLS]']+ tokens_a_new + ['[SEP]']+ tokens_b_new+ ['[SEP]'] , so we can activate new2old_ind[] on it.
        pairs = wnu.find_location_country_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, local_wiki, is_print=False, filter_repeat_word=True)
        # {'alabama': {'location': ['trussville'],
        #              'location_ind': [1],
        #              'country_ind': [0]}}
        offset = (1, offset_new)
        for country in pairs:     # hypo in premise -> country in hypothesis
            for location, location_i in zip(pairs[country]['location'], pairs[country]['location_ind']):
                ext_emb[new2old_ind[location_i + offset[0]]] = 2
                if not (tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0] == location[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0])]):
                    warnings.warn(f"index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0]}' != '{location[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0])]}'")
                    my_logger(f"Warning! index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0]}' != '{location[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[location_i + offset[0]]]])[0])]}'")
                    my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
                    print('********************************************************')
            for country_i in pairs[country]['country_ind']:
                ext_emb[new2old_ind[country_i + offset[1]]] = 1
                if not (tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0] == country[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0])]):
                    my_logger(f"index of wordpiece doesn't match the index of original sentence: '{tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0]}' != '{country[:len(tokenizer.convert_ids_to_tokens([input_ids[new2old_ind[country_i + offset[1]]]])[0])]}'")
                    my_logger(f"\nExample:\nPremise:{example.text_a}\nHypothesis:{example.text_b}\n")
                    print('********************************************************')
    elif ee_gen_method == 'color_wiki_filtered_clean':      # extracting only pairs with Relation=Color.
        s_new, new2old_ind = word_piece_connected(tokens, input_ids, model_type)  # s_new is the connected tokens list.
        tokens_a_new, tokens_b_new = word_piece_connected(tokens_a, [1]*len(tokens_a), model_type)[0], word_piece_connected(tokens_b, [1]*len(tokens_b), model_type)[0]   # connecting the word_pieces together
        offset_new = get_offset_new(tokens_a_new, model_type)  # offset from location in tokens_a_new to location here: ['[CLS]']+ tokens_a_new + ['[SEP]']+ tokens_b_new+ ['[SEP]'] , so we can activate new2old_ind[] on it.
        pairs = wnu.find_color_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, local_wiki_features, is_print=False, filter_repeat_word=True)
        # pairs = {'grey': {'noun': ['steel'],
        #           'noun_ind': [6],
        #           'feature_ind': [9],
        #           'type': 'color'}}
        offset = (1, offset_new)
        for color in pairs:     # hypo in premise -> country in hypothesis
            for noun, noun_i in zip(pairs[color]['noun'], pairs[color]['noun_ind']):
                ext_emb[new2old_ind[noun_i + offset[0]]] = 2
                validate_new_ids(noun, input_ids, noun_i, new2old_ind, offset[0])
            for feature_i in pairs[color]['feature_ind']:
                ext_emb[new2old_ind[feature_i + offset[1]]] = 1
                validate_new_ids(color, input_ids, feature_i, new2old_ind, offset[1])
        pass
    elif ee_gen_method == 'trademark_wiki_filtered_clean':  # extracting only pairs with Relation=Country of Origin (=trademarks).
        s_new, new2old_ind = word_piece_connected(tokens, input_ids, model_type)  # s_new is the connected tokens list.
        tokens_a_new, tokens_b_new = word_piece_connected(tokens_a, [1]*len(tokens_a), model_type)[0], word_piece_connected(tokens_b, [1]*len(tokens_b), model_type)[0]   # connecting the word_pieces together
        offset_new = get_offset_new(tokens_a_new, model_type)  # offset from location in tokens_a_new to location here: ['[CLS]']+ tokens_a_new + ['[SEP]']+ tokens_b_new+ ['[SEP]'] , so we can activate new2old_ind[] on it.
        pairs = wnu.find_trademark_country_pairs(example.text_a, example.text_b, tokens_a_new, tokens_b_new, local_wiki, is_print=False, filter_repeat_word=True)
        # pairs = {'grey': {'noun': ['steel'],
        #           'noun_ind': [6],
        #           'feature_ind': [9],
        #           'type': 'company'}}
        offset = (1, offset_new)
        for country in pairs:     # hypo in premise -> country in hypothesis
            for noun, noun_i in zip(pairs[country]['company'], pairs[country]['company_ind']):
                ext_emb[new2old_ind[noun_i + offset[0]]] = 2
                validate_new_ids(noun, input_ids, noun_i, new2old_ind, offset[0])
            for feature_i in pairs[country]['country_ind']:
                ext_emb[new2old_ind[feature_i + offset[1]]] = 1
                validate_new_ids_for_country(country, country2adj_func(country).lower(), input_ids, feature_i, new2old_ind, offset[1])
        pass
    else:
        raise ValueError("Invalid ee_gen_method:" + ee_gen_method)
    return ext_emb


def debug_first_words(examples):

    def my_tokenizer(str):
        str = str.replace('.', ' . ')
        str = str.replace(',', ' , ')
        str = str.replace('\'', ' \' ')
        str = str.replace('\"', ' \" ')
        str = str.replace('-', ' - ')
        str = str.replace('!', ' ! ')
        return str.strip().split(' ')

    def enhance_hypernyms(words):
        global item2class
        for i, w in enumerate(words):
            if w in item2class:
                w_class = item2class[w]
                if w_class != w:
                    words = words[0:i+1] + ['(','belong', 'to', w_class, ')'] + words[i+1:]
        return words

    for e in examples:
        words_a = my_tokenizer(e.text_a)
        words_b = my_tokenizer(e.text_b)
        # words_a = enhance_hypernyms(words_a)
        # words_b = enhance_hypernyms(words_b)
        if True:
            rand_ind = random.randint(0, len(words_b)-1)
            rand_ind_b = random.randint(0, len(words_b)-1)
            if e.label == 'contradiction':
                words_b[0] = words_a[0] = 'bank'
                # words_b[0] = words_a[0]
                # words_a += 'I love my mom.'
                # words_b += 'I love my mom.'
                # words_b += 'I love mym mom.'
                # words_b[rand_ind] = 'book'
                # words_a[0] = 'book'
                # assert words_a[0] != words_a[2], words_a[2]
            if e.label == 'entailment':
                words_a[0] = 'bank'
                # words_b[0] = words_a[1]
                # words_a += 'I love mym mom.'
                # words_b[rand_ind] = 'hate'
                # words_b[rand_ind] = 'paper'
                # words_a[0] = words_a[2]
                # words_a[0] = 'paper'
                # assert words_a[0] != words_a[1], words_a[1]
            if e.label == 'neutral':
                words_b[0] = 'bank'
                # words_b[0] = words_a[2]
                # words_b += 'I love mym mom.'
                # words_a[rand_ind] = 'hook'
                # pass
                # words_b[rand_ind] = 'table'
                # words_a[1] = words_a[2]
                # words_a[1] = 'table'
                # assert words_a[0] != words_a[1], words_a[0]
        e.text_a = ' '.join(words_a)
        e.text_b = ' '.join(words_b)
    return examples

def comp_models(model_a, model_b, diff_by_module=False):
    model_a_w, module_names = get_model_weights2(model_a)
    model_b_w, _            = get_model_weights2(model_b)
    print_models_dif2(module_names, model_a_w, model_b_w)
    if diff_by_module:
        for key in model_a.state_dict().keys():
            diff = (model_a.state_dict()[key].data - model_b.state_dict()[key].data).abs().mean().item()
            if diff > 0:
                print(f"'{key}' diff = {diff}")

def moduls_var(model):
    for key in model.state_dict().keys():
        v = model.state_dict()[key].data.var().item()
        my_logger("%42s: %1.1e" % (key, v) + 'O'*int(np.log10(v+1e-12) +12)*3)
        # print(key, '\t\t%1.1e'%v)


def get_model_weights2(model):
    layerToDisplay =  args.ext_emb_concat_layer
    weights = [
                copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.query),
                copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.key),
                copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.value),
                copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.output.dense),
                copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layerToDisplay].intermediate.dense),
                copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layerToDisplay]),
                copy.deepcopy(model.module.bert_classifier.bert.embeddings),
                copy.deepcopy(model.module.bert_classifier.classifier),
                copy.deepcopy(model.module.bert_classifier.bert.encoder.knowbert_block),
                copy.deepcopy(model.module.ext_embeddings),
    ]

    layers_weights = [copy.deepcopy(model.module.bert_classifier.bert.encoder.layer[layer]) for layer in range(args.num_hidden_layers)]
    names = ['Query', 'Key', 'Value', 'Norm Linear', 'Intermediate','All layer',
             'BERT Embeddings', 'Classifier', 'KAR', 'Ext Embeddings']
    assert len(weights) == len(names)
    return (weights, layers_weights), names

def print_models_dif2(module_names, prev_model_weights, cur_model_weights):
    print("\n\nprev - currnt*********************************************************")
    prev_weights, prev_layers = prev_model_weights
    cur_weights, cur_layers = cur_model_weights
    for module_name, p, c in zip(module_names, prev_weights, cur_weights):
        print("%22s: %1.0e" % (module_name, model_dif(p, c)), 'O'*int(np.log10(model_dif(p, c)+1e-12) +12)*3)
    print('\nLayers:')
    for layer, (p, c) in enumerate(zip(prev_layers, cur_layers)):
        acc = model_dif(p, c)
        print("              layer %2d: %1.0e" % (layer, acc), 'O'*int(np.log10(acc + 1e-12) +12)*3)
    print('\n')

def model_dif(model_a, model_b):
    tot_diff = []
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        tot_diff.append((pa - pb).abs().mean().item())
    return sum(tot_diff)/len(tot_diff)


def get_iterable_args(args, excluded_loop_args):

    args_out = []
    for arg in vars(args):
        val = getattr(args, arg)
        if type(val) == list and arg not in excluded_loop_args:
            if len(val)>1:
                # print(arg, val)
                args_out.append(arg)
            elif len(val) == 1:                           # converting list arguments with one element into non-list argument.
                setattr(args, arg, val[0])
            elif len(val) == 0:
                setattr(args, arg, None)

    assert len(args_out) <= 2, "You can only have up to two itterable argmunets"

    arg1, vals1, arg2, vals2 = 'do_train', [True], 'do_train', [True]   # initializing these arguments, so in case of no iterations, the loop would be over this argument, and only once.
    if len(args_out) > 0:
        arg1 = args_out[0]
        vals1 = getattr(args, arg1)
    if len(args_out) > 1:
        arg2 = args_out[1]
        vals2 = getattr(args, arg2)
    return arg1, vals1, arg2, vals2


def backward_compatability_loop_variables(args):
    return args.learning_rate, args.max_train_examples, args.max_mix_MNLI_examples, args.seed

def main():

    warnings.filterwarnings("ignore")

    # Forcing parameters
    args.do_lower_case = True
    args.data_dir = '../datasets/MNLI'
    args.cached_features_dir = 'cached_features'
    args.max_seq_length = 128
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.comp_before = True
    args.task_name = 'MNLI'
    args.do_eval = True
    args.temp_models_dir = 'models/temp_models'
    args.teaching_dir = 'teaching_data'
    args.do_save_taught_model = True
    args.no_label_balancing = True

    if not os.path.exists(args.temp_models_dir):
        os.mkdir(args.temp_models_dir)

    if args.templates_dir == None:
        args.templates_dir = 'teaching_data/' + args.EP + '_Templates/auto_generated_templates/'
        print("args.templates_dir is now: ", args.templates_dir)
    if args.data_from_templ_dir == None:
        EP2data_dirs = {
            'Location': '../datasets/inferbert_datasets_separated/Location/',
            'Color': '../datasets/inferbert_datasets_separated/Color/',
            'Trademark': '../datasets/inferbert_datasets_separated/CountryOfOrigin/',
            'Hypernymy': '../datasets/inferbert_datasets_separated/Hypernymy/',
            'Combined': '../datasets/inferbert_datasets_combined',
        }
        if args.EP in EP2data_dirs:
            args.data_from_templ_dir = EP2data_dirs[args.EP]
        # if args.EP == 'Dir_Hypernymy':
        #     args.data_from_templ_dir = 'teaching_data/amturk_data/Hypernymy_datasets/'
        # elif args.EP == 'Location':
        #     # args.data_from_templ_dir = '../teaching_data/amturk_data/Location_datasets/'
        #     args.data_from_templ_dir = 'data/amturk_data/Location_datasets/'
        # elif args.EP == 'Color':
        #     args.data_from_templ_dir = 'data/amturk_data/Color_datasets/'
        else:
            args.data_from_templ_dir = 'teaching_data/' + args.EP + '_Templates/EP_datasets/'
        print("args.templates_dir is now: ", args.templates_dir)
    # args.templates_dir = '../teaching_data/Datives_Templates/auto_generated_templates/'
    # args.data_from_templ_dir = '../teaching_data/Datives_Templates/EP_datasets'
    # args.templates_dir = '../teaching_data/Numbers_Templates/auto_generated_templates/'
    # args.data_from_templ_dir = '../teaching_data/Numbers_Templates/EP_datasets'
    args.mix_teaching_with_MNLI = args.mix_teaching_with_MNLI or (('MNLI' in args.train_setname) if args.train_setname else False)
    args.bert_finetuned_model_dir = "models/BERT_base_84.56"
    args.roberta_finetuned_model_dir = "models/roberta_base_87.0"
    args.lm_pretrained_model_dir = "../BERT_base_pretrained"
    args.pretraining_type = "MNLI"
    args.short_results = True
    assert (args.debug_rule is None) or (args.ee_gen_method is None), "use either debug_rule or ee_gen_method. not both. "
    args.ee_gen_method = args.debug_rule if args.debug_rule is not None else args.ee_gen_method

    assert args.EP in ['Datives','Numbers','Hypernymy','Dir_Hypernymy', 'Location', 'Color', 'Trademark','Combined']
    assert args.pretraining_type in ['MNLI','none']
    assert args.pretraining_type in ['MNLI','none']
    assert args.train_order in ['shuffled', 'MNLI_last', 'MNLI_first'], "train_order must be any of the following: 'shuffled', 'MNLI_last', 'MNLI_first'"
    assert args.train_order == 'shuffled' or args.mix_teaching_with_MNLI == True, "if train_order isn't 'shuffled' then you must have MNLI as part of the training sets"
    assert args.acc_to_compare in ['MNLI_eval', 'S6_cont']
    assert (args.test_group and not args.test_setname) or (not args.test_group and args.test_setname)       # make sure if --test_group entered, so --test_setname is empty.
    assert (args.train_group and not args.train_setname) or (not args.train_group and args.train_setname)   # make sure if --train_group entered, so --train_setname is empty.
    assert args.ext_embeddings_type in ['', 'fixed_added','class_fixed_added', 'first_item_in_class', 'class_auto_added', 'class_auto_concat', 'class_fixed_concat', 'class_fixed_manual_concat', 'first_item_in_class_concat', 'class_auto_project', 'class_auto_attention'], "args.ext_embeddings_type=%s is invalid"%(args.ext_embeddings_type)
    assert args.model_type in ['bert', 'roberta'], "args.model_type=%s is invalid"%(args.model_type)
    assert args.config_ext_emb_method == None, "args.config_ext_emb_method is for internal use only. Do not enter it from the command line"

    if not os.path.exists(args.taught_model_dir):
        os.makedirs(args.taught_model_dir)

    server_name = socket.gethostname()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    wandb_project_name = 'InferBert'

    ## Google Spreadsheet
    global gs
    repository_dir_and_files = ('my_repo',  ['main.py', 'modeling_edited.py','wordnet_parsing_utils.py'])
    already_overwritten_data = []
    already_overwritten_weight = []

    processors = {
        "mnli": MnliProcessor,
    }

    num_labels_task = {
        "mnli": 3,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    template_processor = TemplateProcessor.TemplateProcessor()
    num_labels = num_labels_task[task_name]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case) if args.model_type == 'bert' else RobertaTokenizer.from_pretrained(args.roberta_model)

    def group_to_set(group):
        # group can be 'dative_simple' or 'dative_task2_train[2]'

        dative_all = "T1_build T1_get T1_grant T1_offer T1_provide T2_bring T2_give T2_make T2_send T2_write T1_award T1_buy T1_give T1_lend T1_make T1_send T2_build T2_provide T2_show T1_show T1_bring T1_write T1_lend T2_award T2_buy T2_get T2_grant T2_offer "
        Datives_S = "T1_build T1_get T1_grant T1_offer T1_provide T2_bring T2_give T2_make T2_send T2_write "
        Datives_M = "T1_award T1_buy T1_give T1_lend T1_make T1_send T2_build T2_provide T2_show "
        Datives_C = "T1_show T1_bring T1_write T2_lend T2_award T2_buy T2_get T2_grant T2_offer "

        Datives_n_S = "T1n_build T1n_get T1n_grant T1n_offer T1n_provide T2n_bring T2n_give T2n_make T2n_send T2n_write "
        Dativesn_n_C = "T1n_show T1n_bring T1n_write T2n_lend T2n_award T2n_buy T2n_get T2n_grant T2n_offer "
        Datives_n_S1 = "T2n_bring_1 T2n_give_1 T2n_make_1 T2n_send_1 T2n_write_1 "
        Datives_n_C1 = "T2n_award_1 T2n_buy_1 T2n_get_1 T2n_grant_1 T2n_offer_1 "
        Datives_n_S2 = "T2n_bring_2 T2n_give_2 T2n_make_2 T2n_send_2 T2n_write_2 "
        Datives_n_C2 = "T2n_award_2 T2n_buy_2 T2n_get_2 T2n_grant_2 T2n_offer_2 "
        Datives_n_S2b = "T2n_bring_2b T2n_give_2b T2n_make_2b T2n_send_2b T2n_write_2b "
        Datives_n_C2b = "T2n_award_2b T2n_buy_2b T2n_get_2b T2n_grant_2b T2n_offer_2b "


        Datives_S1 = "T2_bring_1 T2_give_1 T2_make_1 T2_send_1 T2_write_1 "
        Datives_M1 = "T2_build_1 T2_provide_1 T2_show_1 "
        Datives_C1 = "T2_award_1 T2_buy_1 T2_get_1 T2_grant_1 T2_offer_1 "
        # write -> read
        # show -> present
        # send -> offer
        # provide -> prepare
        # offer -> show
        # make -> provide
        # lend -> give
        # grant -> sell
        #  give -> pass
        # get -> find
        # buy -> order
        # build -> make
        # bring -> mail
        # award -> promise
        Datives_S2 = "T2_bring_2 T2_give_2 T2_make_2 T2_send_2 T2_write_2 "
        Datives_M2 = "T2_build_2 T2_provide_2 T2_show_2 "
        Datives_C2 = "T2_award_2 T2_buy_2 T2_get_2 T2_grant_2 T2_offer_2 "

        Datives_S2b = "T2_bring_2b T2_give_2b T2_make_2b T2_send_2b T2_write_2b "
        Datives_M2b = "T2_build_2b T2_provide_2b T2_show_2b "
        Datives_C2b = "T2_award_2b T2_buy_2b T2_get_2b T2_grant_2b T2_offer_2b "


        Hypernymy_S = "S0 S1 S2 S3 S4 "
        Hypernymy_C = "C0 C1 C2 C3 C4 "

        MNLI_and_Hypernymy_S_1A = "MNLI S0_1_A S1_1_A S2_1_A S3_1_A S4_1_A "
        # separating hypernym classes into two groups: A and B
        Hypernymy_S_1A = "S0_1_A S1_1_A S2_1_A S3_1_A S4_1_A "
        Hypernymy_C_1A = "C0_1_A C1_1_A C2_1_A C3_1_A C4_1_A "
        Hypernymy_S_2A = "S0_2_A S1_2_A S2_2_A S3_2_A S4_2_A "
        Hypernymy_C_2A = "C0_2_A C1_2_A C2_2_A C3_2_A C4_2_A "

        Hypernymy_S_2B = "S0_2_B S1_2_B S2_2_B S3_2_B S4_2_B "
        Hypernymy_C_2B = "C0_2_B C1_2_B C2_2_B C3_2_B C4_2_B "

        # separating hypernym items into two groups: IA and IB
        Hypernymy_S_1IA = "S0_1_IA S1_1_IA S2_1_IA S3_1_IA S4_1_IA "
        Hypernymy_C_1IA = "C0_1_IA C1_1_IA C2_1_IA C3_1_IA C4_1_IA "
        Hypernymy_S_2IA = "S0_2_IA S1_2_IA S2_2_IA S3_2_IA S4_2_IA "
        Hypernymy_C_2IA = "C0_2_IA C1_2_IA C2_2_IA C3_2_IA C4_2_IA "

        Hypernymy_S_2IB = "S0_2_IB S1_2_IB S2_2_IB S3_2_IB S4_2_IB "
        Hypernymy_C_2IB = "C0_2_IB C1_2_IB C2_2_IB C3_2_IB C4_2_IB "



        Numbers_S = "S0 S1 S2 S3 S4 S5 S6 S7 S8 "
        Numbers_M = "M0 M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 "
        Numbers_C = "C0 C1 C2 C3 C4 C5 C6 C7 C8 "

        Numbers_S1_R100_199 = ' '.join(['%s_1_R100-199' % s for s in Numbers_S[:-1].split()])
        Numbers_M1_R100_199 = ' '.join(['%s_1_R100-199' % s for s in Numbers_M[:-1].split()])
        Numbers_C1_R100_199 = ' '.join(['%s_1_R100-199' % s for s in Numbers_C[:-1].split()])

        Numbers_S1_R30_50 = ' '.join(['%s_1_R30-50' % s for s in Numbers_S[:-1].split()])
        Numbers_M1_R30_50 = ' '.join(['%s_1_R30-50' % s for s in Numbers_M[:-1].split()])
        Numbers_C1_R30_50 = ' '.join(['%s_1_R30-50' % s for s in Numbers_C[:-1].split()])

        Numbers_S2_R30_50 = ' '.join(['%s_2_R30-50' % s for s in Numbers_S[:-1].split()])
        Numbers_M2_R30_50 = ' '.join(['%s_2_R30-50' % s for s in Numbers_M[:-1].split()])
        Numbers_C2_R30_50 = ' '.join(['%s_2_R30-50' % s for s in Numbers_C[:-1].split()])

        Numbers_S2_R60_80 = ' '.join(['%s_2_R60-80' % s for s in Numbers_S[:-1].split()])
        Numbers_M2_R60_80 = ' '.join(['%s_2_R60-80' % s for s in Numbers_M[:-1].split()])
        Numbers_C2_R60_80 = ' '.join(['%s_2_R60-80' % s for s in Numbers_C[:-1].split()])


        Numbers_S1_R200_299 = ' '.join(['%s_1_R200-299' % s for s in Numbers_S[:-1].split()])
        Numbers_M1_R200_299 = ' '.join(['%s_1_R200-299' % s for s in Numbers_M[:-1].split()])
        Numbers_C1_R200_299 = ' '.join(['%s_1_R200-299' % s for s in Numbers_C[:-1].split()])

        Numbers_S2_R100_199 = ' '.join(['%s_2_R100-199' % s for s in Numbers_S[:-1].split()])
        Numbers_M2_R100_199 = ' '.join(['%s_2_R100-199' % s for s in Numbers_M[:-1].split()])
        Numbers_C2_R100_199 = ' '.join(['%s_2_R100-199' % s for s in Numbers_C[:-1].split()])

        Numbers_S2_R200_299 = ' '.join(['%s_2_R200-299' % s for s in Numbers_S[:-1].split()])
        Numbers_M2_R200_299 = ' '.join(['%s_2_R200-299' % s for s in Numbers_M[:-1].split()])
        Numbers_C2_R200_299 = ' '.join(['%s_2_R200-299' % s for s in Numbers_C[:-1].split()])


        Numbers_S2_R400_499 = ' '.join(['%s_2_R400-499' % s for s in Numbers_S[:-1].split()])
        Numbers_M2_R400_499 = ' '.join(['%s_2_R400-499' % s for s in Numbers_M[:-1].split()])
        Numbers_C2_R400_499 = ' '.join(['%s_2_R400-499' % s for s in Numbers_C[:-1].split()])

        Numbers_S2_R1k_10k = ' '.join(['%s_2_R1k-10k' % s for s in Numbers_S[:-1].split()])
        Numbers_M2_R1k_10k = ' '.join(['%s_2_R1k-10k' % s for s in Numbers_M[:-1].split()])
        Numbers_C2_R1k_10k = ' '.join(['%s_2_R1k-10k' % s for s in Numbers_C[:-1].split()])



        # preparing 5 groups for Datives cross validation
        dative_simple_l = Datives_S.strip().split(' ')*20
        dative_medium_l = Datives_M.strip().split(' ')*20
        dative_complex_l = Datives_C.strip().split(' ')*20
        datives_task2_train, datives_task2_test  = [], []
        datives_task3a_train_S, datives_task3a_test_S, datives_task3a_train_M, datives_task3a_test_M, datives_task3a_train_C, datives_task3a_test_C = [], [], [], [], [], []
        for i in range(5):
            ind0_S = (i*2)
            ind1_S = (i*2 + 7)
            ind2_S = (i*2 + 7 + 2)
            ind0_M = (i*2)
            ind1_M = (i*2 + 7)
            ind2_M = (i * 2 + 7 + 2)
            datives_task2_train_S  = " ".join(dative_simple_l[ind0_S:ind1_S])
            datives_task2_test_S = " ".join(dative_simple_l[ind1_S:ind2_S])
            datives_task2_train_M = " ".join(dative_medium_l[ind0_M:ind1_M])
            datives_task2_test_M = " ".join(dative_medium_l[ind1_M:ind2_M])
            datives_task2_train_C = " ".join(dative_complex_l[ind0_M:ind1_M])
            datives_task2_test_C = " ".join(dative_complex_l[ind1_M:ind2_M])
            datives_task2_train.append(datives_task2_train_S + ' ' + datives_task2_train_M + ' ' + datives_task2_train_C + ' ')
            datives_task2_test.append(datives_task2_test_S + ' ' + datives_task2_test_M + ' ' + datives_task2_test_C + ' ')

            datives_task3a_train_S.append(datives_task2_train_S)
            datives_task3a_test_S.append(datives_task2_test_S)
            datives_task3a_train_M.append(datives_task2_train_M)
            datives_task3a_test_M.append(datives_task2_test_M)
            datives_task3a_train_C.append(datives_task2_train_C)
            datives_task3a_test_C.append(datives_task2_test_C)

        # preparing 5 groups for Numbers cross validation
        numbers_simple_l = Numbers_S.strip().split(' ')*20
        numbers_medium_l = Numbers_M.strip().split(' ')*20
        numbers_complex_l = Numbers_C.strip().split(' ')*20
        numbers_task2_train, numbers_task2_test  = [], []
        numbers_task3a_train_S, numbers_task3a_test_S, numbers_task3a_train_M, numbers_task3a_test_M, numbers_task3a_train_C, numbers_task3a_test_C = [], [], [], [], [], []
        for i in range(5):
            ind0_S = (i*2)
            ind1_S = (i*2 + 7)
            ind2_S = (i*2 + 7 + 2)
            ind0_M = (i*2)
            ind1_M = (i*2 + 7)
            ind2_M = (i * 2 + 7 + 2)
            numbers_task2_train_S  = " ".join(numbers_simple_l[ind0_S:ind1_S])
            numbers_task2_test_S = " ".join(numbers_simple_l[ind1_S:ind2_S])
            numbers_task2_train_M = " ".join(numbers_medium_l[ind0_M:ind1_M])
            numbers_task2_test_M = " ".join(numbers_medium_l[ind1_M:ind2_M])
            numbers_task2_train_C = " ".join(numbers_complex_l[ind0_M:ind1_M])
            numbers_task2_test_C = " ".join(numbers_complex_l[ind1_M:ind2_M])
            numbers_task2_train.append(numbers_task2_train_S + ' ' + numbers_task2_train_M + ' ' + numbers_task2_train_C + ' ')
            numbers_task2_test.append(numbers_task2_test_S + ' ' + numbers_task2_test_M + ' ' + numbers_task2_test_C + ' ')

            numbers_task3a_train_S.append(numbers_task2_train_S)
            numbers_task3a_test_S.append(numbers_task2_test_S)
            numbers_task3a_train_M.append(numbers_task2_train_M)
            numbers_task3a_test_M.append(numbers_task2_test_M)
            numbers_task3a_train_C.append(numbers_task2_train_C)
            numbers_task3a_test_C.append(numbers_task2_test_C)


        # preparing 5 groups for Hypernymy cross validation
        hypernymy_simple_l = Hypernymy_S.strip().split(' ')*20
        hypernymy_complex_l = Hypernymy_C.strip().split(' ')*20
        hypernymy_task2_train, hypernymy_task2_test  = [], []
        hypernymy_task3a_train_S, hypernymy_task3a_test_S, hypernymy_task3a_train_C, hypernymy_task3a_test_C = [], [], [], []
        for i in range(5):
            ind0_S = (i)
            ind1_S = (i + 4)
            hypernymy_task2_train_S  = " ".join(hypernymy_simple_l[ind0_S:ind1_S])
            hypernymy_task2_test_S = hypernymy_simple_l[ind1_S]
            hypernymy_task2_train_C = " ".join(hypernymy_complex_l[ind0_S:ind1_S])
            hypernymy_task2_test_C = hypernymy_complex_l[ind1_S]
            hypernymy_task2_train.append(hypernymy_task2_train_S + ' '  + hypernymy_task2_train_C + ' ')
            hypernymy_task2_test.append(hypernymy_task2_test_S + ' '  + hypernymy_task2_test_C + ' ')

            hypernymy_task3a_train_S.append(hypernymy_task2_train_S)
            hypernymy_task3a_test_S.append(hypernymy_task2_test_S)
            hypernymy_task3a_train_C.append(hypernymy_task2_train_C)
            hypernymy_task3a_test_C.append(hypernymy_task2_test_C)

        return eval(group)

    def init_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def init_torch_seed(seed):
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def get_start_stamp():
        st1 = 'Pretrained on MNLI' if args.pretraining_type=='MNLI' else 'No Pretraining'
        st23 = '(max % d examples)' % max_mix_MNLI_examples if max_mix_MNLI_examples >= 0 else ''
        st22 = ' Mixed with MNLI %s' % (st23) if args.mix_teaching_with_MNLI and st23  else ''
        st3 = '(max %d examples)' % max_train_examples if max_train_examples> -1 else ''
        test_group_s = ('Test Group: ' + test_group) if test_group else ''
        train_group_s = ('Train Group: ' + train_group + ' ') if train_group else ''
        start_stamp = '\n----- Time: %s ; Server: %s Loop %d---------------------------------------------------------------------------------------------\n' % (start_time, server_name, loop_counter)
        start_stamp += ' '.join(sys.argv) + '\n\n'
        start_stamp += 'Start_runtime: ' + start_runtime_s + '\n'
        start_stamp += '1) %s\n' % st1
        start_stamp += '2) Before: Eval on %s ;  %s ;Test sets: %s (max %d examples)\n' % (args.task_name, test_group_s, test_setname_str, args.max_test_examples)
        start_stamp += '3) Training on %sTraining sets: %s  %s %s, lr = %1.1e\n' % (train_group_s, train_setname_str, st3, st22, args.learning_rate)
        if args.mix_teaching_with_MNLI:
            start_stamp += '   Training set order: %s + %s\n' % (args.train_order, 'RandomSampler' if args.semi_random_train else 'SequentialSampler')
        start_stamp += '   Seed: %s\n' % seed
        return start_stamp

    def get_loop_params():
        if len(args.max_train_examples)>1 and len(args.learning_rate)<=1 and len(args.max_mix_MNLI_examples)<=1 and len(args.seed)<=1:
            max_train_examples = args.max_train_examples
            lr = args.learning_rate * len(max_train_examples)
            max_mix_MNLI_examples = args.max_mix_MNLI_examples * len(max_train_examples)
            seed = args.seed * len(lr)
            loop_param = 'max_train_examples'
            loop_val_list = args.max_train_examples
        elif len(args.learning_rate)>1 and len(args.max_train_examples)<=1 and len(args.max_mix_MNLI_examples)<=1 and len(args.seed)<=1:
            lr = args.learning_rate
            max_train_examples = args.max_train_examples * len(lr)
            max_mix_MNLI_examples = args.max_mix_MNLI_examples * len(lr)
            seed = args.seed * len(lr)
            loop_param = 'lr'
            loop_val_list = args.learning_rate
        elif len(args.max_mix_MNLI_examples)>1 and len(args.learning_rate)<=1 and len(args.max_train_examples)<=1 and len(args.seed)<=1 :
            max_mix_MNLI_examples = args.max_mix_MNLI_examples
            lr = args.learning_rate * len(max_mix_MNLI_examples)
            max_train_examples = args.max_train_examples * len(max_mix_MNLI_examples)
            seed = args.seed * len(lr)
            loop_param = 'max_mix_MNLI_examples'
            loop_val_list = args.max_mix_MNLI_examples
        elif len(args.seed)>1 and len(args.max_mix_MNLI_examples)<=1 and len(args.learning_rate)<=1 and len(args.max_train_examples)<=1:
            seed = args.seed
            max_mix_MNLI_examples = args.max_mix_MNLI_examples * len(args.seed)
            lr = args.learning_rate * len(args.seed)
            max_train_examples = args.max_train_examples * len(args.seed)
            loop_param = 'seed'
            loop_val_list = args.seed
        elif len(args.max_train_examples)==1 and len(args.learning_rate)==1 and len(args.max_mix_MNLI_examples)==1 and len(args.seed)==1:
            lr = args.learning_rate
            max_train_examples = args.max_train_examples
            max_mix_MNLI_examples = args.max_mix_MNLI_examples
            seed = args.seed
            loop_param = None
            loop_val_list = None
        else:
            raise ImportError("Learning_rate and max_train_examples can't be both multiple values")
        return lr, max_train_examples, max_mix_MNLI_examples, seed, loop_param, loop_val_list


    def prepare_dev_examples():
        # In case of Dev is only MNLI (dev-matched)
        if args.dev_setname == ['MNLI']:
            my_logger("Loading MNLI dev matched as the dev set.")
            mnli_eval_examples = processor.get_dev_examples(args.data_dir)
            mnli_eval_examples = mnli_eval_examples[:args.max_mnli_eval_examples]
            my_logger('Number of NLI dev examples: %d' % len(mnli_eval_examples), 1)

            # loading dummy dev_templates as a programming shortcut
            dev_setname = 'combined_dev'
            _, dev_templates_t = template_processor.get_templates(args.templates_dir, args.EP, dev_setname, args.data_from_templ_dir, is_label_balancing=False) # no need for label balancing in dev
            return mnli_eval_examples, dev_templates_t


        dev_examples, dev_templates = [], []
        max_example_per_set = args.max_dev_examples//len(args.dev_setname)    # dividing max_dev_examples evenly between the sets
        my_logger("max_example_per_set = %d (max_dev_examples=%d / number of dev sets=%d)" % (max_example_per_set, args.max_dev_examples, len(args.dev_setname)))
        for dev_setname in args.dev_setname:
            my_logger("get template from dev set %s" % (dev_setname))
            dev_examples_t, dev_templates_t = template_processor.get_templates(args.templates_dir, args.EP, dev_setname, args.data_from_templ_dir, is_label_balancing=False) # no need for label balancing in dev
            # Sampling only max_example_per_set from each set:
            if max_example_per_set > 0 and max_example_per_set < len(dev_examples_t):
                my_logger("Randomizing only %d examples from total of %d dev examples of '%s'" % (max_example_per_set, len(dev_examples_t), dev_setname))
                ind = sorted(random.sample(range(len(dev_examples_t)), k=max_example_per_set))
                dev_examples_t = [dev_examples_t[i] for i in ind]
            else:
                my_logger("Taking all examples (%d) from the dev set %s" % (len(dev_examples_t), dev_setname))
            dev_examples += dev_examples_t
            dev_templates += dev_templates_t
        my_logger(">>>>> Total Dev examples: %d <<<<<" % (len(dev_examples)))
        wan.update({'#Dev_Examples': len(dev_examples)})
        return dev_examples, dev_templates


    def prepare_test_examples():
        test_examples, test_templates = [], []
        max_example_per_set = args.max_test_examples//len(args.test_setname)    # dividing max_test_examples evenly between the sets
        my_logger("max_example_per_set = %d (max_test_examples=%d / number of test sets=%d)" % (max_example_per_set, args.max_test_examples, len(args.test_setname)))
        for test_setname in args.test_setname:
            my_logger("get template from test set %s" % (test_setname))
            test_examples_t, test_templates_t = template_processor.get_templates(args.templates_dir, args.EP, test_setname, args.data_from_templ_dir, is_label_balancing=False) # no need for label balancing in test
            # Sampling only max_example_per_set from each set:
            if max_example_per_set > 0 and max_example_per_set < len(test_examples_t):
                my_logger("Randomizing only %d examples from total of %d test examples of '%s'" % (max_example_per_set, len(test_examples_t), test_setname))
                ind = sorted(random.sample(range(len(test_examples_t)), k=max_example_per_set))
                test_examples_t = [test_examples_t[i] for i in ind]
            else:
                my_logger("Taking all examples (%d) from the test set %s" % (len(test_examples_t), test_setname))
            test_examples += test_examples_t
            test_templates += test_templates_t
        my_logger(">>>>> Total Test examples: %d <<<<<" % (len(test_examples)))
        wan.update({'#Test_Examples': len(test_examples)})
        return test_examples, test_templates


    def prepare_training_examples():
        global num_train_optimization_steps

        num_train_optimization_steps = None
        train_examples, train_templates = [], []

        ## MNLI is the training set itself (and only MNLI)
        if args.train_setname == ['MNLI']:
            my_logger("--- Preparing Synthetic examples:")
            my_logger("Loading MNLI as the training set.")
            MNLI_train_examples = processor.get_train_examples(args.data_dir)
            random.shuffle(MNLI_train_examples)
            MNLI_train_examples = MNLI_train_examples[:max_mix_MNLI_examples]
            my_logger(f"loaded {len(MNLI_train_examples)} MNLI training examples")

            # getting dummy templates
            train_setname = 'combined_train'       # dummy existing training set
            _, train_templates_t = template_processor.get_templates(args.templates_dir, args.EP, train_setname, args.data_from_templ_dir, is_label_balancing=not args.no_label_balancing)

            num_train_optimization_steps = int(len(MNLI_train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            if args.local_rank != -1:
                num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
            wan.update({'#Train_Examples': len(MNLI_train_examples)})
            return MNLI_train_examples, train_templates_t

        # Getting Training Set examples. E.g. S6_2
        my_logger("--- Preparing Synthetic examples:")
        nb_sets = len(args.train_setname)-1 if 'MNLI' in args.train_setname else len(args.train_setname)
        max_example_per_set = max_train_examples // nb_sets  # dividing max_train_examples evenly between the training sets
        for train_setname in args.train_setname:
            if train_setname.upper()!='MNLI':       # in case we have MNLI as one of the sets in the arguments string:  --training_set S5 MNLI
                train_examples_t, train_templates_t = template_processor.get_templates(args.templates_dir, args.EP, train_setname, args.data_from_templ_dir, is_label_balancing=not args.no_label_balancing)
                # Sampling only max_example_per_set from each set:
                label_balance_s = '(after label balancing)' if (not args.no_label_balancing) else ''
                if max_example_per_set >= 0 and max_example_per_set < len(train_examples_t):
                    if not args.random_sampling_training_set:
                        my_logger("Pseudo-randomizing only %d examples from total of %d training examples of %s %s" % (max_example_per_set, len(train_examples_t), train_setname, label_balance_s))
                        train_examples_t = ut.semiRandomSample(train_examples_t, max_example_per_set)
                    else:
                        my_logger("Randomizing only %d examples from total of %d training examples of %s %s" % (max_example_per_set, len(train_examples_t), train_setname, label_balance_s))
                        ind = random.sample(range(len(train_examples_t)), k=max_example_per_set)
                        train_examples_t = [train_examples_t[i] for i in ind]
                    if max_example_per_set == 0:
                        my_logger("Warning: max_example_per_set = 0")
                else:
                    my_logger("Taking all examples (%d) from training set %s %s" % (len(train_examples_t), train_setname, label_balance_s))
                train_examples += train_examples_t
                train_templates += train_templates_t
        if args.duplicate_train_examples > 1:
            train_examples = train_examples * args.duplicate_train_examples

        # Adding MNLI examples
        if args.mix_teaching_with_MNLI:
            my_logger("--- Preparing MNLI examples:")
            my_logger("Reading MNLI examples...")
            MNLI_train_examples = processor.get_train_examples(args.data_dir)
            st_add = ", but adding only the first %d examples to the training set." % max_mix_MNLI_examples if max_mix_MNLI_examples >= 0 else '.'
            my_logger('Fininshed reading %d MNLI training set from %s%s' % (len(MNLI_train_examples), args.data_dir, st_add))
            if max_mix_MNLI_examples >= 0:      # taking only the first max_mix_MNLI_examples from the complete MNLI training set
                MNLI_train_examples = MNLI_train_examples[args.mnli_examples_start_ind:args.mnli_examples_start_ind+max_mix_MNLI_examples]
                my_logger('Taking MNLI examples %d .. %d'%(args.mnli_examples_start_ind, args.mnli_examples_start_ind+max_mix_MNLI_examples))
            random.shuffle(MNLI_train_examples)

            # Ordering training set (MNLI first, last, or shuffled with my training set)
            if args.train_order == 'MNLI_last':
                my_logger("------\ntrain_examples = [Synthetic (%d)] + [MNLI (%d)]"% (len(train_examples), len(MNLI_train_examples)))
                combined_train_examples = train_examples + MNLI_train_examples
            elif args.train_order == 'MNLI_first':
                my_logger("------\ntrain_examples = [MNLI (%d)] + [Synthetic (%d)]"% (len(MNLI_train_examples), len(train_examples)))
                combined_train_examples = MNLI_train_examples + train_examples
            elif args.train_order == 'shuffled':
                my_logger("------\ntrain_examples = shuffle ([Synthetic (%d)] + [MNLI (%d)])"% (len(train_examples), len(MNLI_train_examples)))
                combined_train_examples = train_examples + MNLI_train_examples
                random.shuffle(combined_train_examples)
        else:
            combined_train_examples = train_examples
            my_logger(">>>>> Total Training examples (not including MNLI): %d <<<<<" % (len(train_examples)))

        num_train_optimization_steps = int(len(combined_train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        wan.update({'#Train_Examples': len(combined_train_examples)})
        return combined_train_examples, train_templates

    def manage_freezing(args, model, freeze_type):
        ## BERT
        if args.model_type == 'bert' and args.config_ext_emb_method is not None:    # Don't freeze anything if there are no external embeddings
            if freeze_type == 'all_but_ext_emb':  # freeze all model, except ext_embeddings
                # Freezing model
                freeze_weights(model.module.bert_classifier)
                # Unfreezing ext_embeddings (and parts of QKV for concat if needed)
                freeze_weights(model.module.ext_embeddings, defreeze=True)
                if args.config_ext_emb_method == 'concat':
                    freeze_weights(model.module.bert_classifier.bert.encoder.layer[args.ext_emb_concat_layer].attention.self, ['query_t', 'key_t', 'value_t'], defreeze=True)
                if args.config_ext_emb_method.lower() == 'kar':
                    freeze_weights(model.module.bert_classifier.bert.encoder.knowbert_block, defreeze=True)
            elif freeze_type == 'ext_emb':  # freeze only external embeddings, and unfreeze rest of model.
                # Unfreezing model
                freeze_weights(model.module.bert_classifier, defreeze=True)
                freeze_weights(model.module.ext_embeddings)
            elif freeze_type == 'ext_emb_and_kar':  # freeze external embeddings and KAR, and unfreeze rest of model.
                # Unfreezing model
                freeze_weights(model.module.bert_classifier, defreeze=True)
                # Freezing ext_embeddings (and parts of QKV for concat if needed)
                freeze_weights(model.module.ext_embeddings)
                if args.config_ext_emb_method == 'concat':
                    freeze_weights(model.module.bert_classifier.bert.encoder.layer[args.ext_emb_concat_layer].attention.self, ['query_t', 'key_t', 'value_t'])
                if args.config_ext_emb_method.lower() == 'kar':
                    freeze_weights(model.module.bert_classifier.bert.encoder.knowbert_block)
            elif (freeze_type is None):
                freeze_weights(model.module, defreeze=True)
            elif freeze_type == 'user_input':
                # Unfreezing model
                freeze_weights(model.module, defreeze=True)
                if args.config_ext_emb_method == 'concat':
                    freeze_weights(model.module.bert_classifier.bert.encoder.layer[args.ext_emb_concat_layer].attention.self, ['query_t', 'key_t', 'value_t'], defreeze=True)
                if args.config_ext_emb_method == 'kar':
                    freeze_weights(model.module.bert_classifier.bert.encoder.knowbert_block.kar_attention.multihead, ['query', 'key', 'value'], defreeze=True)
            else:
                raise ValueError("Invalid freeze_type")

            ## bypassing freeze_type (the input argument)
            if args.freeze_bert:  # freezing BERT model (but leaving the classifier and ext_embeddings trainable)
                freeze_weights(model.module.bert_classifier.bert.encoder)
            if args.freeze_ext_emb_and_kar:  # freezing external embeddings weights and KAR's
                freeze_weights(model.module.ext_embeddings)
                if args.config_ext_emb_method == 'kar':
                    # freeze_weights(model.module.bert_classifier.bert.encoder.knowbert_block.kar_attention.multihead, ['query', 'key', 'value'])
                    freeze_weights(model.module.bert_classifier.bert.encoder.knowbert_block.kar_attention.multihead, ['query', 'key', 'value'])
            if args.freeze_ext_emb:  # freezing external embeddings weights
                freeze_weights(model.module.ext_embeddings)

        ## RoBERTa
        elif args.model_type == 'roberta' and args.config_ext_emb_method is not None: # Don't freeze anything if there are no external embeddings
            if freeze_type == 'all_but_ext_emb':  # step 1: freeze all model, except ext_embeddings
                # Freezing model
                freeze_weights(model.module.roberta_classifier)
                # Unfreezing ext_embeddings (and parts of QKV for concat if needed)
                freeze_weights(model.module.ext_embeddings, defreeze=True)
                if args.config_ext_emb_method == 'concat':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.layer[args.ext_emb_concat_layer].attention.self, ['query_t', 'key_t', 'value_t'], defreeze=True)
                if args.config_ext_emb_method.lower() == 'kar':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.knowbert_block, defreeze=True)
            elif freeze_type == 'ext_emb':  # freeze only external embeddings, and unfreeze rest of model.
                # Unfreezing model
                freeze_weights(model.module.roberta_classifier, defreeze=True)
                freeze_weights(model.module.ext_embeddings)
            elif freeze_type == 'ext_emb_and_kar':  # freeze external embeddings and KAR, and unfreeze rest of model.
                # Unfreezing model
                freeze_weights(model.module.roberta_classifier, defreeze=True)
                # Freezing ext_embeddings (and parts of QKV for concat if needed)
                freeze_weights(model.module.ext_embeddings)
                if args.config_ext_emb_method == 'concat':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.layer[args.ext_emb_concat_layer].attention.self, ['query_t', 'key_t', 'value_t'])
                if args.config_ext_emb_method.lower() == 'kar':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.knowbert_block)
            elif (freeze_type is None):
                freeze_weights(model.module, defreeze=True)
            elif freeze_type == 'user_input':
                # Unfreezing model
                freeze_weights(model.module, defreeze=True)
                if args.config_ext_emb_method == 'concat':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.layer[args.ext_emb_concat_layer].attention.self, ['query_t', 'key_t', 'value_t'], defreeze=True)
                if args.config_ext_emb_method == 'kar':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.knowbert_block.kar_attention.multihead, ['query', 'key', 'value'], defreeze=True)
            else:
                raise ValueError("Invalid freeze_type")

            ## bypassing freeze_type (the input argument)
            if args.freeze_bert:  # freezing BERT model (but leaving the classifier and ext_embeddings trainable)
                freeze_weights(model.module.roberta_classifier.roberta.encoder)
            if args.freeze_ext_emb_and_kar:  # freezing external embeddings weights and KAR's
                freeze_weights(model.module.ext_embeddings)
                if args.config_ext_emb_method == 'kar':
                    freeze_weights(model.module.roberta_classifier.roberta.encoder.knowbert_block.kar_attention.multihead, ['query', 'key', 'value'])
            if args.freeze_ext_emb:  # freezing external embeddings weights
                freeze_weights(model.module.ext_embeddings)

    def train_model(model, train_examples, optimizer, lr=args.learning_rate, freeze_type=None, nb_epochs=3):
        global global_step

        if train_examples == []:
            return model, -1, -1, '', ''

        # Sampling a small set from train_examples for train accuracy check
        if args.train_acc_size < len(train_examples):
            ind = sorted(random.sample(range(len(train_examples)), k=args.train_acc_size))
            train_examples_for_acc = [train_examples[i] for i in ind]
        else:
            train_examples_for_acc = train_examples

        train_dataloader = load_or_gen_features('train', train_examples)

        model.train()
        manage_freezing(args, model, freeze_type)

        epoch = 0
        eval_acc_by_label_str = ''
        dist_by_label_str =''
        training_results = []
        freeze_list = []    # used to 'manually' freeze sub_modules that don't freeze using requires_grad=False due to momentum
        for _ in trange(int(nb_epochs), desc="Epoch"):
            my_logger('\n\n\n------------ Epoch %d -----------'%epoch)
            if freeze_type == 'user_input':
                freeze_list, is_early_finish = get_interactive_freeze(args.freeze_input[epoch])
                if is_early_finish: break
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            if args.print_trained_layers:
                prev_model_weights, module_names = get_model_weights2(model)  # for debug only
            prev_par = copy_model_params(freeze_list)   # used to 'manually' freeze sub_modules that don't freeze using requires_grad=False due to momentum
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                if args.model_type=='bert':
                    input_ids, input_mask, segment_ids, label_ids, ext_emb_ids, example_weight, example_source = batch
                    loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, ext_emb_ids=ext_emb_ids, example_weight=example_weight, fix_position=args.fix_position)
                elif args.model_type=='roberta':
                    input_ids, input_mask, label_ids, ext_emb_ids, example_weight, example_source = batch   # same like 'bert' but without 'segment_ids'
                    outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids, ext_emb_ids=ext_emb_ids, example_weight=example_weight, fix_position=args.fix_position)
                    loss = outputs[0]

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = lr * WarmupLinearSchedule(global_step / num_train_optimization_steps,
                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    # force_freeze(freeze_list)
                    global_step += 1

                    # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #     my_logger('loss: %1.2f'%((tr_loss - logging_loss) / args.logging_steps))
                    #     # tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    #     logging_loss = tr_loss

            paste_model_params(prev_par, freeze_list) # used to 'manually' freeze sub_modules that don't freeze using requires_grad=False due to momentum
            epoch_loss = tr_loss / nb_tr_steps
            my_logger('Epoch loss: %1.2f' % epoch_loss)
            del prev_par

            if args.print_trained_layers:   # debug. track the changes of weights of the different modules.
                cur_model_weights, _ = get_model_weights2(model)  # for debug only
                print_models_dif2(module_names, prev_model_weights, cur_model_weights)

            if args.do_eval_train:
                my_logger('\n\t\t\t Evaluating Training set and MNLI accuracy after Epoch %d:\n' % epoch)
                train_acc, eval_loss, eval_acc_by_label_str, dist_by_label_str = eval_examples_batch(train_examples_for_acc, model, data_type='train_eval')
                train_acc_str = " Train: %1.2f" % train_acc * 100
                mnli_acc,_ ,_, _,_ = get_MNLI_dev_acc(args.data_dir, model)
                my_logger('-------\nTraining set accuracy for Epoch %d: %s, MNLI Dev:= %1.2f\n'%(epoch, train_acc_str, mnli_acc*100))

            if args.test_during_train or (freeze_type =='all_but_ext_emb' and args.num_of_rand_init>1):
                my_logger('\nEvaluating Dev set accuracy after Epoch %d:\n' % epoch)
                dev_acc, eval_loss, eval_acc_by_label_str, dist_by_label_str = eval_examples_batch(dev_examples, model)
                my_logger('-------\nDev set accuracy for epoch %d=%1.2f%% %s, loss=%1.2f \n%s\n'%(epoch, dev_acc*100, eval_acc_by_label_str, eval_loss, dist_by_label_str))
                wan.update({'epoch %d acc'%(epoch):'%1.2f%%'%(dev_acc*100)})
                training_results.append((epoch_loss, eval_loss, dev_acc))
            epoch +=1
        if args.test_during_train or (freeze_type =='all_but_ext_emb' and args.num_of_rand_init > 1):
            print_epochs_data(training_results)
            last_eval_loss = eval_loss
            last_dev_acc = dev_acc
        else:
            last_eval_loss = -1
            last_dev_acc = -1
        del loss
        torch.cuda.empty_cache()
        return model, last_eval_loss, last_dev_acc, eval_acc_by_label_str, dist_by_label_str

    def print_epochs_data(training_results):
        epoch_losses_s = str([round(x[0],2) for x in training_results])
        test_losses_s = str([round(x[1],2) for x in training_results])
        dev_accs = str([round(x[2]*100,1) for x in training_results])
        my_logger('\n\nTraining Results: \n\tEpoch losses: %s\n\tTest losses: %s\n\tTest accs: %s'%(epoch_losses_s, test_losses_s, dev_accs))

    def print_models_dif(module_names, prev_model_weights, cur_model_weights):
        print("\n\nprev - currnt*********************************************************")
        prev_weights, prev_layers = prev_model_weights
        cur_weights, cur_layers = cur_model_weights
        for module_name, p, c in zip(module_names, prev_weights, cur_weights):
            print("%22s: %1.0e" % (module_name, (p - c).abs().mean()), 'O'*int(np.log10((p - c).abs().mean().item()+1e-12) +12)*3)
        print('\nLayers:')
        for layer, (prev_layer, cur_layer) in enumerate(zip(prev_layers, cur_layers)):
            acc = 0
            for p, c in zip(prev_layer, cur_layer):
                acc += (p - c).abs().mean()
            acc /= len(prev_layer)
            print("              layer %2d: %1.0e" % (layer, acc), 'O'*int(np.log10(acc.item()+1e-12) +12)*3)
        print('\n')








    def find_models_dict_diffs(model_a, model_b):
        my_logger("\n\nprev - currnt*********************************************************")
        for module in model_a:
            w_a = model_a[module]
            w_b = model_b[module]
            diff = (w_a - w_b).abs().mean()
            if diff >0:
                my_logger('%s: %1.1e'%(module, diff))

    def force_freeze(model, prev_modules, mudules_to_freeze):
        for module in mudules_to_freeze:
            module.data =1


    def get_model_weights(model):
        layerToDisplay =  args.ext_emb_concat_layer
        weights = [
                    model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.query.weight.data.clone(),
                    model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.key.weight.data.clone(),
                    model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.value.weight.data.clone(),
                    ## (next 3 lines are dupricated)
                    # model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.query_t.weight.data.clone(),
                    # model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.key_t.weight.data.clone(),
                    # model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.self.value_t.weight.data.clone(),
                    model.module.bert_classifier.bert.encoder.layer[layerToDisplay].attention.output.dense.weight.data.clone(),
                    model.module.bert_classifier.bert.encoder.layer[layerToDisplay].intermediate.dense.weight.data.clone(),
                    model.module.bert_classifier.bert.embeddings.word_embeddings.weight.data.clone(),
                    model.module.bert_classifier.bert.embeddings.position_embeddings.weight.data.clone(),
                    model.module.bert_classifier.bert.embeddings.token_type_embeddings.weight.data.clone(),
                    model.module.bert_classifier.classifier.weight.data.clone(),
                    model.module.bert_classifier.bert.encoder.knowbert_block.kar_attention.multihead.query.weight.data.clone(),
                    model.module.ext_embeddings.weight.data.clone(),
        ]

        layers_weights = [(
            model.module.bert_classifier.bert.encoder.layer[layer].attention.self.query.weight.data.clone(),
            model.module.bert_classifier.bert.encoder.layer[layer].attention.self.key.weight.data.clone(),
            model.module.bert_classifier.bert.encoder.layer[layer].attention.self.value.weight.data.clone(),
            ## (next 3 lines are dupricated)
            # model.module.bert_classifier.bert.encoder.layer[layer].attention.self.query_t.weight.data.clone(),
            # model.module.bert_classifier.bert.encoder.layer[layer].attention.self.key_t.weight.data.clone(),
            # model.module.bert_classifier.bert.encoder.layer[layer].attention.self.value_t.weight.data.clone(),
            model.module.bert_classifier.bert.encoder.layer[layer].attention.output.dense.weight.data.clone(),
            model.module.bert_classifier.bert.encoder.layer[layer].intermediate.dense.weight.data.clone() )
            for layer in range(args.num_hidden_layers)]
        names = ['Query', 'Key', 'Value', 'Norm Linear', 'Intermediate',
                 'Word Embeddings', 'Position Embeddings', 'Token Type Embeddings', 'Classifier', 'KAR', 'Ext Embeddings']
        assert len(weights) == len(names)
        return (weights, layers_weights), names


    def copy_model_params(sub_model_list):
        # receives a list of sub_modules to copy
        par = None
        for sub_model in sub_model_list:
            par = [p.clone() for p in sub_model.parameters()]
        return par

    def paste_model_params(source_par, sub_model_list):
        # receives a list of sub_modules to copy into an existing module
        for sub_model in sub_model_list:
            for p, prevp in zip(sub_model.parameters(), source_par):
                p.data = prevp.data.clone()

    def get_interactive_freeze(user_in=None):
        print("\n-------------> Please enter freeze typ (1- attention, 2- intermediate, 3- bert embeddings, 4- MLP class., 5- Ext. Emb: ")
        if user_in is None:
            user_in = input()
        freeze_list = []
        freeze_weights(model.module)    # freeze all, and then defreeze by user request
        if '1' in user_in:    # train SelfAttention
            print("--- Training SelfAttention")
            freeze_weights(model.module.bert_classifier.bert.encoder.layer[0].attention, defreeze=True)
        if '2' in user_in:    # train Intermediate
            print("--- Training Intermediate")
            freeze_weights(model.module.bert_classifier.bert.encoder.layer[0].intermediate, defreeze=True)
        if '3' in user_in:    # train bert embeddings
            print("--- Training bert embeddings")
            freeze_weights(model.module.bert_classifier.bert.embeddings, defreeze=True)
        if '4' in user_in:    # train MLP of classifier
            print("--- Training MLP of classifier")
            freeze_weights(model.module.bert_classifier.classifier, defreeze=True)
        if '5' in user_in:    # train external embeddings
            print("--- Training external embeddings")
            freeze_weights(model.module.ext_embeddings, defreeze=True)
        if '6' in user_in:    # train Intermediate
            print("--- Training layer 0")
            freeze_weights(model.module.bert_classifier.bert.encoder.layer[0], defreeze=True)
            freeze_list = [model.module.bert_classifier.bert.encoder.layer[1]]
        if '7' in user_in:    # train Intermediate
            print("--- Training layer 1")
            freeze_weights(model.module.bert_classifier.bert.encoder.layer[1], defreeze=True)
            freeze_list = [model.module.bert_classifier.bert.encoder.layer[0]]
        if user_in in ['', 'all']:
            freeze_weights(model.module, defreeze=True)
        is_early_finish = user_in.lower() == 'end'
        return freeze_list, is_early_finish


    def save_model(model, model_dir, model_name):
        """ Save a trained model and the associated configuration """
        output_model_file = os.path.join(model_dir, model_name)
        my_logger('Saving model to %s' % output_model_file, 1)
        torch.save(model.state_dict(), output_model_file)
        output_config_file = os.path.join(model_dir, NEW_CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model.module.config.to_json_string())

    def add_to_config(config, args):
        if args.ext_embeddings_type in ['fixed_added', 'class_fixed_added', 'first_item_in_class', 'class_auto_added']:
            config.ext_emb_size = config.hidden_size
            config.ext_emb_method = 'add'
        elif args.ext_embeddings_type in ['class_auto_concat','class_fixed_manual_concat']:
            config.ext_emb_size = args.concat_embeddings_size   # default: 10
            config.ext_emb_method = 'concat'
            # args.two_steps_train_with_freeze = True
        elif args.ext_embeddings_type in ['class_fixed_concat', 'first_item_in_class_concat']:
            config.ext_emb_size = config.hidden_size   # default: 10
            config.ext_emb_method = 'concat'
            # args.two_steps_train_with_freeze = True
        elif args.ext_embeddings_type in ['class_auto_project']:
            config.ext_emb_size = args.concat_embeddings_size  # default: 10
            config.ext_emb_method = 'project'
            # args.two_steps_train_with_freeze = True
        elif args.ext_embeddings_type in ['class_auto_attention']:
            config.ext_emb_size = args.kar_ext_emb_size  # default: 96 (changed to 768)
            config.ext_emb_method = 'kar'
            # args.two_steps_train_with_freeze = True
        elif args.ext_embeddings_type == '':
            config.ext_emb_size = config.hidden_size
            config.ext_emb_method = None
        else:
            raise Exception ("Invalid ext_embeddings_type")

        # if args.force_no_freeze:
            # args.two_steps_train_with_freeze = False

        config.ext_vocab_size = 1000
        config.ext_emb_concat_layer = args.ext_emb_concat_layer
        config.norm_ext_emb = args.norm_ext_emb
        config.num_hidden_layers = args.num_hidden_layers
        config.uniform_ext_emb = args.uniform_ext_emb
        config.no_kar_norm = args.no_kar_norm
        config.debug_type = args.model_debug_type

        args.config_ext_emb_method = config.ext_emb_method      # for internal use only (for to be added by commandline).

        return config

    def init_and_load_bert_classifier_model(model_dir, model_name):
        load_model_file = os.path.join(model_dir, model_name)
        load_config_file = os.path.join(model_dir, BERT_CONFIG_NAME)
        my_logger("Loading fine-tuned / pre-trained BERT model %s..."%load_model_file)
        config = BertConfig(load_config_file)
        config = add_to_config(config, args)
        model = BertWrapper(config, load_model_file, num_labels=num_labels, ext_embeddings_type=args.ext_embeddings_type)
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=[0])
        return model

    def init_and_load_roberta_classifier_model(model_dir, model_name):
        load_model_file = os.path.join(model_dir, model_name)
        load_config_file = os.path.join(model_dir, NEW_CONFIG_NAME)
        my_logger("Loading fine-tuned / pre-trained RoBERTa model %s..."%load_model_file)
        config = RobertaConfig(load_config_file)
        config = add_to_config(config, args)
        print("************ config *****************\n", config)
        model = RobertaWrapper(config, load_model_file, num_labels=num_labels, ext_embeddings_type=args.ext_embeddings_type)
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        return model

    def init_random_model(model_dir):
        #TODO: change BERT_CONFIG_NAME to match the new config file type
        load_config_file = os.path.join(model_dir, BERT_CONFIG_NAME)
        my_logger("Initializing (random weights) BERT model...")
        config = BertConfig(load_config_file)
        config = add_to_config(config, args)
        model = BertWrapper(config, None, num_labels=num_labels, ext_embeddings_type=args.ext_embeddings_type)
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=[0])
        return model

    def prepare_optimizer(model, lr = args.learning_rate):
        # Prepare optimizer
        global global_step
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lr,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        global_step = 0

        return optimizer


    def load_or_gen_features(data_type, examples):
        # loads cached features or converts the features from examples.
        # Input:
        #   examples: training or eval examples
        #   data_type: 'train', 'eval' or 'mnli'

        assert data_type in ['train', 'dev', 'test', 'mnli'], "data_type must be either 'train' or 'dev' or 'mnli'"
        dataloader = None   # to make sure it's always declared

        # Setting cached features Filename
        cached_features_file = os.path.join(args.cached_features_dir, 'cached_{}_{}_{}_{}_{}{}{}'.format(
            args.model_type,
            data_type,
            args.ee_gen_method,
            train_setname_str if data_type=='train' else 'mnli' if data_type=='mnli' else dev_setname_str if data_type=='dev' else test_setname_str,
            str(max_train_examples) if data_type=='train' else str(args.max_mnli_eval_examples) if data_type=='mnli' else str(args.max_dev_examples) if data_type=='dev' else str(args.max_test_examples),
            ('_maxMNLI_%d'%(max_mix_MNLI_examples)) if (max_mix_MNLI_examples >= 0 and data_type=='train') else '',
            ('_w*%d'%args.weight_to_challenge_set),
        ))
        if not os.path.exists(args.cached_features_dir):
            os.mkdir(args.cached_features_dir)

        overwrite = args.overwrite_cached_features and (data_type not in already_overwritten_data or args.weight_to_challenge_set not in already_overwritten_weight)
        if os.path.exists(cached_features_file) and not overwrite:
            my_logger("Loading %s features from cached file %s" % (data_type, cached_features_file))
            features = torch.load(cached_features_file)
            my_logger(f"Loaded {len(features)} features")
        else:
            my_logger("Creating %s features from dataset file at %s" % (data_type, cached_features_file))
            features = convert_examples_to_features(args, examples, args.max_seq_length, tokenizer, args.ext_embeddings_type, args.break_position, data_type)
            my_logger(f"Created {len(features)} features")
            my_logger("Saving %s features into cached file %s"%(data_type, cached_features_file))
            torch.save(features, cached_features_file)
            already_overwritten_data.append(data_type)
            already_overwritten_weight.append(args.weight_to_challenge_set)

        if args.model_type == 'bert':
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_ext_emb_ids = torch.tensor([f.ext_emb_ids for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            all_example_weight_ids = torch.tensor([f.example_weight for f in features], dtype=torch.float)
            all_example_source_ids = torch.tensor([f.example_source for f in features], dtype=torch.long)
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ext_emb_ids, all_example_weight_ids, all_example_source_ids)
        elif args.model_type == 'roberta':      # all_segment_ids is excluded in RoBERTa
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_ext_emb_ids = torch.tensor([f.ext_emb_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            all_example_weight_ids = torch.tensor([f.example_weight for f in features], dtype=torch.float)
            all_example_source_ids = torch.tensor([f.example_source for f in features], dtype=torch.long)
            data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_ext_emb_ids, all_example_weight_ids, all_example_source_ids)
        if data_type == 'train':
            if not args.semi_random_train:
                train_sampler = RandomSampler(data)
            else:
                train_sampler = SequentialSampler(data)   # I changed it so I can locate my training examples before or after MNLI, while I shuffled them already
            dataloader = DataLoader(data, sampler=train_sampler, batch_size=args.train_batch_size)
        elif data_type in ['dev', 'test', 'mnli']:
            eval_sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        return dataloader


    def get_MNLI_dev_acc(data_dir, model) :
        my_logger('Loading dev examples from %s...'%data_dir)
        mnli_eval_examples = processor.get_dev_examples(data_dir)
        mnli_eval_examples = mnli_eval_examples[:args.max_mnli_eval_examples]
        my_logger('Number of NLI dev examples: %d'%len(mnli_eval_examples), 1)

        eval_dataloader = load_or_gen_features('mnli', mnli_eval_examples)

        model.eval()
        eval_loss, mnli_acc = 0, 0
        label_count, eval_accuracy_by_label, pred_dist, dist_by_label = np.zeros([3,]), np.zeros([3,]), np.zeros([3,]), np.zeros([9,])
        nb_eval_steps, nb_eval_examples = 0, 0
        results = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if args.model_type == 'bert':
                    input_ids, input_mask, segment_ids, label_ids, ext_emb_ids, example_weight, example_source = batch
                    logits = model(input_ids, segment_ids, input_mask, labels=None, ext_emb_ids=ext_emb_ids, fix_position=args.fix_position)
                elif args.model_type == 'roberta':
                    input_ids, input_mask, label_ids, ext_emb_ids, example_weight, example_source = batch  # same like 'bert' but without 'segment_ids'
                    outputs = model(input_ids, attention_mask=input_mask, labels=None, ext_emb_ids=ext_emb_ids, example_weight=example_weight, fix_position=args.fix_position)
                    logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            example_source = example_source.detach().cpu().numpy()
            pred_label = np.argmax(logits,1)
            probs = np.max(softmax(logits,1), 1)
            tmp_eval_accuracy = accuracy(logits, label_ids)
            tmp_eval_accuracy_by_label, tmp_dist_by_label, _ =  accuracy_by_label(logits, label_ids, example_source)
            temp_pred_dist = pred_distribution(logits)
            label_count += [np.sum(label_ids==0), np.sum(label_ids==1), np.sum(label_ids==2)]
            results += list(zip(pred_label, probs))

            mnli_acc += tmp_eval_accuracy
            eval_accuracy_by_label += tmp_eval_accuracy_by_label
            dist_by_label += tmp_dist_by_label
            pred_dist += temp_pred_dist

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        mnli_acc = mnli_acc / nb_eval_examples
        eval_accuracy_by_label = eval_accuracy_by_label / label_count
        dist_by_label = dist_by_label / ([label_count[0]]*3 + [label_count[1]]*3 + [label_count[2]]*3) # dividing by :[nb_label_0, nb_label_0, nb_label_0, nb_label_1, nb_label_1, nb_label_1, nb_label_2, nb_label_2 nb_label_2]
        pred_dist = 1.* pred_dist / nb_eval_examples
        # MNLI_results.append(mnli_acc)

        my_logger('MNLI dev Accuracy = %1.2f (Acc by label: Cont: %1.1f, Ent: %1.1f, Neut: %1.1f; \t\tPred Dist:  %1.0f%%, %1.0f%%, %1.0f%%)\n'% (mnli_acc*100, eval_accuracy_by_label[0]*100, eval_accuracy_by_label[1]*100, eval_accuracy_by_label[2]*100,pred_dist[0]*100, pred_dist[1]*100, pred_dist[2]*100))
        mnli_eval_acc_by_label_str = "(Acc by label: Cont: %1.1f, Ent: %1.1f, Neut: %1.1f; \t\tPred Dist:  %1.0f%%, %1.0f%%, %1.0f%%)"%(eval_accuracy_by_label[0]*100, eval_accuracy_by_label[1]*100, eval_accuracy_by_label[2]*100,pred_dist[0]*100, pred_dist[1]*100, pred_dist[2]*100)
        dist_by_label_str = "(MNLI Dist by label: **Cont**: Cont:%1.2f, Ent: %1.2f, Neut: %1.2f; \t**Ent**: Cont:%1.2f, Ent: %1.2f, Neut: %1.2f; \t**Neu**: Cont:%1.2f, Ent: %1.2f, Neut: %1.2f; )"%(dist_by_label[0]*100, dist_by_label[1]*100, dist_by_label[2]*100, dist_by_label[3]*100, dist_by_label[4]*100, dist_by_label[5]*100, dist_by_label[6]*100, dist_by_label[7]*100, dist_by_label[8]*100)
        return mnli_acc, results, mnli_eval_examples, mnli_eval_acc_by_label_str, dist_by_label_str


    def eval_examples_batch(eval_examples, model, data_type=None):
        my_logger('\nEvaluating Test Set(%d exapmles):'%(len(eval_examples)))
        data_type = data_type if data_type is not None else 'dev'
        eval_dataloader = load_or_gen_features(data_type, eval_examples)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_count, eval_accuracy_by_label, pred_dist, dist_by_label, eval_accuracy_by_source, source_count = np.zeros([3,]), np.zeros([3,]), np.zeros([3,]), np.zeros([9,]), np.zeros([2,]), np.zeros([2,])
        # results = []
        ind = 0
        eval_loss = 0

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if args.model_type == 'bert':
                    input_ids, input_mask, segment_ids, label_ids, ext_emb_ids, example_weight, example_source = batch
                    tmp_eval_loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, ext_emb_ids=ext_emb_ids, example_weight=example_weight, fix_position=args.fix_position)
                    logits = model(input_ids, segment_ids, input_mask, labels=None, ext_emb_ids=ext_emb_ids, fix_position=args.fix_position)
                elif args.model_type == 'roberta':
                    input_ids, input_mask, label_ids, ext_emb_ids, example_weight, example_source = batch  # same like 'bert' but without 'segment_ids'
                    tmp_outputs = model(input_ids, attention_mask=input_mask, labels=label_ids, ext_emb_ids=ext_emb_ids, example_weight=example_weight, fix_position=args.fix_position)    # output = tupple(loss)
                    tmp_eval_loss = tmp_outputs[0]
                    outputs = model(input_ids, attention_mask=input_mask, labels=None, ext_emb_ids=ext_emb_ids, example_weight=example_weight, fix_position=args.fix_position)           # output = tupple(logits)
                    logits = outputs[0]

            tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()


            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            pred_label = np.argmax(logits,1)
            probs = np.max(softmax(logits,1), 1)*100

            example_source = example_source.detach().cpu().numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            tmp_eval_accuracy_by_label, tmp_dist_by_label, tmp_eval_accuracy_by_source = accuracy_by_label(logits, label_ids, example_source)
            temp_pred_dist = pred_distribution(logits)
            label_count += [np.sum(label_ids==0), np.sum(label_ids==1), np.sum(label_ids==2)]
            source_count += [np.sum(example_source==0), np.sum(example_source==1)]

            # results += list(zip(pred_label, probs))
            eval_accuracy += tmp_eval_accuracy
            eval_accuracy_by_label += tmp_eval_accuracy_by_label
            eval_accuracy_by_source += tmp_eval_accuracy_by_source
            dist_by_label += tmp_dist_by_label
            pred_dist += temp_pred_dist

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            ind += input_ids.shape[0]

        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_accuracy_by_label = eval_accuracy_by_label / label_count
        eval_accuracy_by_source = eval_accuracy_by_source / source_count
        dist_by_label = dist_by_label / ([label_count[0]]*3 + [label_count[1]]*3 + [label_count[2]]*3) # dividing by :[nb_label_0, nb_label_0, nb_label_0, nb_label_1, nb_label_1, nb_label_1, nb_label_2, nb_label_2 nb_label_2]
        pred_dist = 1.* pred_dist / nb_eval_examples

        tot_eval_loss = eval_loss / nb_eval_steps

        eval_acc_by_label_str = "(Acc by label: Cont: %1.2f, Ent: %1.2f, Neut: %1.2f; \t\tPred Dist: %1.2f%%, %1.2f%%, %1.2f%% \tChallenge-exmp Acc: %1.2f%%, MNLI-exmp Acc:%1.2f%%)"%(eval_accuracy_by_label[0]*100, eval_accuracy_by_label[1]*100, eval_accuracy_by_label[2]*100,pred_dist[0]*100, pred_dist[1]*100, pred_dist[2]*100, eval_accuracy_by_source[0], eval_accuracy_by_source[1])
        dist_by_label_str = "(Dist by label: **Cont**: Cont:%1.2f, Ent: %1.2f, Neut: %1.2f; \t**Ent**: Cont:%1.2f, Ent: %1.2f, Neut: %1.2f; \t**Neu**: Cont:%1.2f, Ent: %1.2f, Neut: %1.2f; )"%(dist_by_label[0]*100, dist_by_label[1]*100, dist_by_label[2]*100, dist_by_label[3]*100, dist_by_label[4]*100, dist_by_label[5]*100, dist_by_label[6]*100, dist_by_label[7]*100, dist_by_label[8]*100)
        return eval_accuracy, tot_eval_loss, eval_acc_by_label_str, dist_by_label_str


    def save_train_templates(train_templates):
        info_filename = os.path.join(args.taught_model_dir, 'training_info.txt')
        with open(info_filename, 'w') as f:
            f.write('Trained on: %s \n\n\n' % train_setname_str)
            f.write('Saved on ' + str(datetime.datetime.now()) +'\n')
            f.write(' '.join(sys.argv) + '\n\n')
            for line in template2str(train_templates):
                f.write(str(line) + '\n')

        pkl_filename = os.path.join(args.taught_model_dir, 'train_templates.pkl')
        with open(pkl_filename , 'wb') as handle:
            pickle.dump(train_templates, handle)


    def get_item2class():
        classes = {
            'fruits': ['berries', 'apples', 'bananas', 'oranges', 'lemons', 'peaches', 'grapes', 'pineapples', 'pears', 'watermelons'],
            'vegetables': ['carrots', 'celery', 'cucumbers', 'peppers', 'broccoli', 'peas', 'potatoes', 'lettuce', 'onions', 'spinach'],
            'vehicles': ['ambulances', 'bicycles', 'boats', 'trains', 'buses', 'cars', 'planes', 'scooters', 'helicopters', 'motorcycles'],
            'flowers': ['roses', 'lilies', 'daisies', 'jasmines', 'sunflowers', 'orchids', 'irises', 'tulips', 'gladioli', 'carnations'],
            'trees': ['cottonwoods', 'gums', 'maples', 'palms', 'pines', 'redwoods', 'willows', 'yews', 'birches', 'firs'],
            'mammals': ['pandas', 'dogs', 'cats', 'cows', 'elephants', 'lions', 'tigers', 'bears', 'camels', 'goats'],
            'insects': ['beetles', 'ants', 'flies', 'bees', 'grasshoppers', 'bytterflies', 'crickets', 'termites', 'bugs', 'mosquitoes'],
            'tools': ['hammers', 'pliers', 'pocketknifes', 'handsaws', 'ladders', 'irons', 'jigsaws', 'drills', 'saws', 'knifes'],
            'clothes': ['jeans', 'shorts', 'shirts', 'hoodies', 'sweaters', 'pants', 'skirts', 'tights', 'underwear', 'socks'],
        }

        classes_rare = {
            'fruits': ['ackees', 'rambutans', 'physalises', 'jabuticabas', 'durians', 'mangosteens', 'cherimoyas', 'cupuacus'],
            'vegetables': ['kohlrabis', 'jicamas', 'dulses', 'ramps', 'fiddleheads', 'dulses', 'salsifys', 'sunchokes'],
            'flowers': ['franklins', 'kokios', 'juliets', 'slippers', 'middlemists', 'cosmoses', 'kadupuls', 'corpses'],
            'trees': ['dentells', 'dragonbloods', 'abarkuhs', 'osmanias', 'hibakujumokus', 'bodhis', 'sugis', 'chankiris'],
        }

        item2class = {}
        class2ind = {}
        for cls in classes:
            class2ind[cls] = len(class2ind) +1
            for item in classes[cls]:
                item2class[item] = cls
            if cls in classes_rare:
                for item in classes_rare[cls]:
                    item2class[item] = cls
            item2class[cls] = cls

        return item2class, class2ind

    def results_to_googlesheet(loop_results, loop_MNLI_results, loop_best_acc_results, ):
        status_str = 'Done' if len(loop_val_list) ==1 else 'Evaluated (%d/%d)' % (loop_counter, nb_loops)
        # updating results in Google Sheet
        col_to_update = {'Status': status_str}
        lr_or_not_str = '%1.0e' if loop_param in ['learning_rate', 'ext_learning_rate'] else '%d'
        if len(loop_val_list) > 1:   # there is a loop over a loop argument
            # 'loop param'    |'Loop 1' | 'Res 1' | 'Loop 2' | 'Res 2' | ...
            # 'learning_rate' |'2e-5'    | '93.4%' | '1e-6'    | '95.1%" | ...
            acc = loop_results[loop_counter-1]
            mnli_acc = loop_MNLI_results[loop_counter-1]
            if loop_counter == 1:
                col_to_update.update({'Loop Param': loop_param})
            val = loop_val_list[loop_counter - 1]
            param_val_s = '%1.0e' % val if loop_param  in ['learning_rate', 'ext_learning_rate'] else '%d' % val
            results_s = (loop_param + ': ' + ' '.join([('[' + lr_or_not_str + ': %1.1f%%/%1.1f%%]') % (loop_val_list[i], res * 100, mnli_res * 100) for i, (res, mnli_res, best_acc) in enumerate(zip(loop_results, loop_MNLI_results, loop_best_acc_results))]))
            col_to_add = {'Loop %d' % (loop_counter): param_val_s, 'Res %d' % (loop_counter): '%1.2f%%' % (acc * 100), 'MNLI %d' % (loop_counter): '%1.2f%%' % (mnli_acc * 100),
                          'Results': results_s}
        else:
            col_to_add = {'Results': '%1.2f%%'%(result*100), 'MNLI Results':'%1.2f%%'%(loop_MNLI_results[0]*100)}
        col_to_update.update(col_to_add)
        col_to_update.update({'Dev Acc': ', '.join(['%1.2f'%(res*100) for res in loop_results])})
        col_to_update.update({'MNLI Acc': ', '.join(['%1.2f'%(mnli_res*100) for mnli_res in loop_MNLI_results])})
        col_to_update.update({'Loop_val': ', '.join([lr_or_not_str %(val) for val in loop_val_list[:len(loop_results)]])})
        wan.update(col_to_update)


    def results_to_wan(loop_results, loop_MNLI_results, loop_best_acc_results, test_acc, best_ext_acc, best_dev_acc, best_ext_lr, best_lr, finetuned_model_name, mnli_accuracy):
        status_str = 'Done' if len(loop_val_list) ==1 else 'Evaluated (%d/%d)' % (loop_counter, nb_loops)
        # updating results in Google Sheet
        col_to_update = {'Status': status_str}
        lr_or_not_str = '%1.0e' if loop_param in ['learning_rate', 'ext_learning_rate'] else '%d'

        col_to_add = {'Results': '%1.2f/%1.2f, Best Dev:%1.2f, Best ext:%1.2f' % (test_acc*100, mnli_accuracy*100, best_dev_acc*100, best_ext_acc*100),
                      'MNLI Results':'%1.2f%%'%(mnli_accuracy*100),
                      'HyperParams Results:': 'Ext_LR:%1.0e, LR:%1.0e' % (best_ext_lr, best_lr),
                      'Model Name': finetuned_model_name}

        col_to_update.update(col_to_add)
        col_to_update.update({'Dev Acc': ', '.join(['%1.2f'%(res*100) for res in loop_results])})
        col_to_update.update({'MNLI Acc': ', '.join(['%1.2f'%(mnli_res*100) for mnli_res in loop_MNLI_results])})
        col_to_update.update({'Loop_val': ', '.join([lr_or_not_str %(val) for val in loop_val_list[:len(loop_results)]])})

        wan.update(col_to_update)


    def results_only_test_to_wan(test_acc, mnli_accuracy):
        status_str = 'Done. Test Results: %1.2f%% / %1.2f%%' % (test_acc*100, mnli_accuracy*100)
        # updating results in Google Sheet
        col_to_update = {'Status': status_str}
        col_to_add = {'Results': '%1.2f/%1.2f - Exp %s - Test Only' % (test_acc*100, mnli_accuracy*100, args.inferbert_to_load),
                      'MNLI Results':'%1.2f%% - Test Only'%(mnli_accuracy*100)}

        col_to_update.update(col_to_add)
        wan.update(col_to_update)


    def acc_hist(acc_list):
        # gets accuracy list and calculates the ratio of accs that are: >50%, >70%, >80%, >90%
        res =[]
        thrs = [.70, .80, .90]
        for thr in thrs:
            res.append(len([1 for i in acc_list if i > thr]) * 1. / len(acc_list) * 100)
        return f"len={len(acc_list)}|" + '; '.join([f">{thr*100:.0f}%:{r:.1f}%" for r, thr in zip(res, thrs)])


    def manage_init_and_load_model(args):
        if args.pretrained_model_to_load == 'bert_lm':
            my_logger('\n\n------------------  Loading pretrained BERT LM ------------\n(%s)' % (ut.now_str()) )
            # model = prepare_empty_model_for_finetuning(args.lm_pretrained_model_dir)  # Load pretrained BERT LM (not finetuned on anything)
            # model.to(device)
            model = init_and_load_bert_classifier_model(args.lm_pretrained_model_dir, WEIGHTS_NAME)
        elif args.pretrained_model_to_load == 'bert_mnli':
            my_logger('\n\n------------------  Loading BERT finetuned on MNLI------------\n')
            model = init_and_load_bert_classifier_model(args.bert_finetuned_model_dir, WEIGHTS_NAME)  # Load Finetuned BERT on MNLI
        elif args.pretrained_model_to_load == 'roberta_mnli':
            my_logger('\n\n------------------  Loading Roberta finetuned on MNLI------------\n')
            model = init_and_load_roberta_classifier_model(args.roberta_finetuned_model_dir, WEIGHTS_NAME)  # Load Finetuned BERT on MNLI
        elif args.pretrained_model_to_load == None:  # initializing BERT without pre-training at all
            my_logger('\n\n------------------  Initializing BERT with random weights ------------\n')
            model = init_random_model(args.lm_pretrained_model_dir)  # Initializing (random) full model (BERT + Ext Embeddings)
        else:
            # Todo: I load two models in the next 4 lines. fix it to be only one? Is there a redundancy?
            my_logger('\n\n------------------  Loading existing model ------------\n')
            model_filename = os.path.join(args.taught_model_dir, args.pretrained_model_to_load, FINE_TUNED_NAME)
            model = init_and_load_bert_classifier_model(args.bert_finetuned_model_dir, WEIGHTS_NAME)  # Load Finetuned BERT on MNLI
            my_logger("Loading model %s"%model_filename)
            model.load_state_dict(torch.load(model_filename), strict=False)
        return model

    def train_ext_emb_multiple_inits(i_grid):
        """
        Initializing the model (each time with different random S-KAR weights) and training only the external modules - multiple times (args.num_of_rand_init),
        eventually taking the best model with lowest eval_loss.
        """

        eval_losses = []
        dev_accs = []
        best_i_init = 0
        best_result = 0

        for i_init in range(args.num_of_rand_init):  # initializing model and training external modules multiple times, taking the best model with highest eval_loss
            if args.num_of_rand_init > 1:
                my_logger('\n\n------------------  Random initialization %d/%d ------------\n' % (i_init, args.num_of_rand_init))
            if 'model' in locals(): del model
            model = manage_init_and_load_model(args)  # Initializing and loading model according to arg.pretrained_model_to_load


            if args.test_before:  # Debug: test accuracy after loading
                test_acc, mnli_acc = eval_dataset_and_mnli(model, 'test_before', 'test')
                status_str = 'Test Before: Test Results: %1.2f%% / %1.2f%%' % (test_acc * 100, mnli_acc * 100)
                col_to_update = {'Status': status_str}
                col_to_add = {'Results Before': '(%s) %1.2f/%1.2f - Exp %s - Test Only' % (args.model_type, test_acc * 100, mnli_acc * 100, args.pretrained_model_to_load),
                              'MNLI Results Before': '(%s) %1.2f%% - Test Only' % (args.model_type, mnli_acc * 100)}
                col_to_update.update(col_to_add)
                wan.update(col_to_update)
                exit(0)

            ## training external modules
            my_logger('\n\n ******* Training External Embeddings Only with ext_learning_rate=%1.0e, for %d epochs... **********'%(args.ext_learning_rate, args.num_train_epochs_ext) if i_init == 0 else '')
            optimizer = prepare_optimizer(model, args.ext_learning_rate)
            model, eval_loss, last_dev_acc, dev_acc_by_label_str, dist_by_label_str = train_model(model, train_examples, optimizer, args.ext_learning_rate, freeze_type='all_but_ext_emb', nb_epochs=args.num_train_epochs_ext)  # train only external embeddings
            eval_losses.append(eval_loss)
            eval_losses_s = ' '.join(['%1.2f' % e for e in eval_losses])
            # MNLI evaluation
            mnli_accuracy, _, _, mnli_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)
            dev_accs.append((last_dev_acc, mnli_accuracy))
            dev_accs_s = ',  '.join([' %1.0f/%1.0f' % (a1 * 100, a2 * 100) for (a1, a2) in dev_accs])   # dev acc / MNLI acc
            my_logger(" For init %d\n \tAcc dev: %1.2f%% \t\t\t%s\n" % (i_init, last_dev_acc * 100, dev_acc_by_label_str) + '\t%s\n\tMNLI acc after ext_emb: %1.2f%% \t\t%s\n\t\t%s' % (dist_by_label_str, mnli_accuracy * 100, mnli_acc_by_label_str, mnli_dist_by_label_str), highlight=('@',20))

            my_logger("\n---------------\ni_init = %d, eval_loss = %1.2f" % (i_init, eval_loss))
            best_i_init = np.array([x[0] for x in dev_accs]).argmax()

            if args.save_every_init:        # instead of saving only when the current model is superior to those of all previous inits (as implemented below)- here we save the model EVERY init
                model_to_save = 'exp_%d_%s_init_%d_i_grid_%d_%s_MNLI_%s.bin' % (wan.exp_id, server_name, i_init, i_grid, ("%1.0f" % (last_dev_acc * 100)).replace('.', '_'), ("%1.0f" % (mnli_accuracy * 100)).replace('.', '_'))
                my_logger('Saving best-so-far model to: %s' % model_to_save)
                torch.save(model.state_dict(), os.path.join(args.temp_models_dir, model_to_save))

            if best_i_init == i_init:
                best_eval_loss = eval_loss
                best_acc = last_dev_acc
                acc_s = ("%1.2f" % (best_acc * 100)).replace('.', '_')
                model_to_save = 'exp_%d_%s_init_%d_best_i_grid_%d_%s.bin' % (wan.exp_id, server_name, i_init, i_grid, acc_s)
                my_logger('Saving best-so-far model to: %s' % model_to_save)
                torch.save(model.state_dict(), os.path.join(args.temp_models_dir, model_to_save))
            my_logger('so far, best loss is %1.2f of i_init=%d, best acc=%1.1f%%' % (min(eval_losses), best_i_init, best_acc * 100))
            wan.update({'Ext Rand Losses': eval_losses_s, 'Ext Rand Accs': dev_accs_s, 'Acc Hist': acc_hist([x[0] for x in dev_accs])}, status='Init %d/%d, acc=%1.2f%% (i_grid %d), best i=%d with loss=%1.2f, ' % (i_init, args.num_of_rand_init, best_acc * 100, i_grid, best_i_init, best_eval_loss,))

        my_logger('='*34 + f'Finished {args.num_of_rand_init} inits ' + '='*34, color='blue')

        best_i_init = np.array([x[0] for x in dev_accs]).argmax()
        my_logger("Dev losses = %s"%eval_losses_s, color='blue')
        my_logger("Dev Accs = %s"%dev_accs_s, color='blue')
        acc_s = ("%1.2f" % (dev_accs[best_i_init][0]*100)).replace('.', '_')
        model_to_load = 'exp_%d_%s_init_%d_best_i_grid_%d_%s.bin' % (wan.exp_id, server_name, best_i_init, i_grid, acc_s)
        my_logger("Loading best model of i= %d (should be: loss=%1.2f, acc=%1.2f%%)  		filename: %s"%(best_i_init, eval_losses[best_i_init],dev_accs[best_i_init][0]*100, model_to_load), color='blue')
        model.load_state_dict(torch.load(os.path.join(args.temp_models_dir, model_to_load)), strict=False)
        # TODO -remove next line (sanity check - making sure this is the correct model loaded):
        result, _, acc_by_label_str, dist_by_label_str = eval_examples_batch(dev_examples, model)
        my_logger("Acc Test Result (after loading): %1.2f%% %s\n\n"%(result*100, acc_by_label_str), color='blue')
        my_logger("Acc Test Result (after loading) by label:%s\n\n"%(dist_by_label_str), color='blue')
        if result!=best_acc:
            my_logger("\n\n!!!!!! BUG: Best result: %1.2f%%, loaded result: %1.2f%%\n\n"%(best_acc*100, result*100), color='blue')

        # MNLI Acc
        my_logger("----------------------- Evaluating MNLI set after finished training ext_emb: -----------------------", color='blue')
        mnli_accuracy, eval_after, eval_examples, mnli_eval_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)
        my_logger("\n--- External Embeddings Results: ---\n \tAcc Test Result (after loading): %1.2f%% %s\n" % (result * 100, acc_by_label_str) + '\tMNLI acc after ext_emb: %1.2f%% %s\n\t%s\n\t%s' % (mnli_accuracy * 100, mnli_eval_acc_by_label_str, mnli_dist_by_label_str, mnli_dist_by_label_str), highlight=('@', 84), color='blue')

        # Saving best model
        model_to_save = 'exp_%d_ext_emb.bin' % (wan.exp_id)
        my_logger('Saving best model to: %s' % model_to_save, color='blue')
        torch.save(model.state_dict(), os.path.join(args.temp_models_dir, model_to_save))

        return model, best_acc, mnli_accuracy

    def load_existing_inferbert_model(model_name):
        my_logger('\n\n------------------  Loading existing model ------------\n')
        model_filename_full = os.path.join(args.taught_model_dir, model_name, FINE_TUNED_NAME)
        model = init_and_load_bert_classifier_model(args.bert_finetuned_model_dir, WEIGHTS_NAME)  # Load Finetuned BERT on MNLI
        my_logger("Loading model %s" % model_filename_full)
        model.load_state_dict(torch.load(model_filename_full), strict=False)
        return model

    def save_taught_model(model):
        file_path = os.path.join(args.taught_model_dir, str(wan.exp_id) + ('_' + str(loop_counter) if loop_counter>1 else ''))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        my_logger("---- Saving the fine-fine-tuned model at %s ---" % (file_path))
        save_model(model, file_path, FINE_TUNED_NAME)
        save_train_templates(train_templates)

    def eval_dataset_and_mnli(model, training_step, dev_or_test):
        title = {'bert': 'after training BERT only (freezing S-KAR)', 'all': 'after training ALL weights', 'test_before': 'Right after loading', 'ext_emb': '', 'kar':'after training KAR & ExtEmb only'}
        ## Evaluating Test (or Dev) Set
        my_logger('Testing %s:'%(title[training_step]), color='red')
        examples_to_evaluate = dev_examples if dev_or_test == 'dev' else test_examples if dev_or_test == 'test' else None
        eval_acc, _, eval_acc_by_label_str, dist_by_label_str = eval_examples_batch(examples_to_evaluate, model, data_type=dev_or_test)
        set_s = dev_or_test[0].upper() + dev_or_test[1:]
        my_logger(set_s + ' set acc %s: %1.2f%%' % (title[training_step], eval_acc * 100), color='red')
        my_logger(set_s + " dist by label:%s\n\n" % (dist_by_label_str), color='red')
        ## Evaluating MNLI
        my_logger("Evaluating MNLI set:", color='red')
        mnli_accuracy, eval_after, eval_examples, mnli_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)
        my_logger('MNLI acc %s: %1.2f%% %s' % (title[training_step], mnli_accuracy * 100, mnli_acc_by_label_str), color='red')
        my_logger("%s\n \tTest acc: %1.2f%% \t\t\t%s\n\t\t\t\t\t\t%s\n" % (title[training_step], eval_acc * 100, eval_acc_by_label_str, dist_by_label_str) + '\tMNLI acc: %1.2f%% \t\t%s\n\t\t\t\t\t\t%s' % (mnli_accuracy * 100, mnli_acc_by_label_str, mnli_dist_by_label_str), highlight=('@', 20), color='red')
        return eval_acc, mnli_accuracy

    def eval_only_mnli(model):
        mnli_acc, eval_after, eval_examples, _, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)
        my_logger('MNLI acc: %1.2f%% %s' % (mnli_acc * 100, eval_acc_by_label_str))
        my_logger('MNLI dist by label: %s' % (mnli_dist_by_label_str))

    def save_model_finetuning(model, i, i_name, dev_acc, mnli_acc):
        acc_s = ("%1.2f" % (dev_acc * 100)).replace('.', '_')
        model_to_save = 'exp_%d_%s_%s_best_%d_loop_%d_%s.bin' % (wan.exp_id, server_name, i_name, i, loop_counter, acc_s)
        my_logger('Saving Grid best-so-far model to: %s' % model_to_save)
        torch.save(model.state_dict(), os.path.join(args.temp_models_dir, model_to_save))

    def fineture_ext_weights():
        """ Scans the ext_learning_rate (grid search; according to args.ext_learning_rate_vec) to find the learning rate for the S_KAR weights according to dev results"""

        dev_accs = []

        for i_grid, ext_lr in enumerate(args.ext_learning_rate_vec):
            args.ext_learning_rate = ext_lr
            if len(args.ext_learning_rate_vec) > 1:
                my_logger('\n\n==================  Grid Search  Ext-LR=%1.1e (%d/%d) ===============\n' % (ext_lr, i_grid, len(args.ext_learning_rate_vec)-1), color='magenta')

            ## Training external modules (S-KAR layer)
            model, init_dev_acc, mnli_acc = train_ext_emb_multiple_inits(i_grid)    # initializing the model (multiple times with different seeds) and training only the external modules - multiple times, taking the best model with highest eval_loss
            dev_accs.append((init_dev_acc, mnli_acc))
            dev_accs_s = ',\t'.join([' %1.2f/%1.2f' % (a1 * 100, a2 * 100) for (a1, a2) in dev_accs])   # dev acc / MNLI acc
            dev_accs_s_wan = ''.join(['[%1.0e: %1.1f/%1.1f]' % (e, a1 * 100, a2 * 100) for e, (a1, a2) in zip(args.ext_learning_rate_vec, dev_accs)])   # dev acc / MNLI acc
            my_logger("For ext_lr %1.1e\n \tAcc dev: %1.2f%%\n" % (ext_lr, init_dev_acc*100) + '\n\tMNLI acc after ext_emb: %1.2f%%'%(mnli_acc*100), highlight=('=@', 20), color='magenta')
            wan.update({'Grid Ext-LR %d acc' % (i_grid): '%1.2f%%/%1.2f%%' % (init_dev_acc*100, mnli_acc*100)})


            best_i_grid = np.array([x[0] for x in dev_accs]).argmax()

            if args.save_every_grid_step:        # instead of saving only when the current model is superior to those of all previous inits (as implemented below)- here we save the model EVERY init
                save_model_finetuning(model, i_grid, 'ext_lr_grid', init_dev_acc, mnli_acc)

            if best_i_grid == i_grid:
                best_acc = init_dev_acc
                save_model_finetuning(model, i_grid, 'ext_lr_grid', best_acc, mnli_acc)

            my_logger('so far, best acc=%1.1f%% for i_grid=%d, ' % (best_acc * 100, best_i_grid), color='magenta')
            wan.update({'Ext-LR Grid Accs': dev_accs_s_wan})

        my_logger('='*60 + f'Finished Grid Searches for {len(args.ext_learning_rate_vec)} External Learning Rates' + '='*60, color='green')

        best_i_grid = np.array([x[0] for x in dev_accs]).argmax()
        best_ext_lr = args.ext_learning_rate_vec[best_i_grid]
        wan.update({'Best Ext-LR': f"{best_ext_lr:1.0e}| Acc={dev_accs[best_i_grid][0]:.2f}, MNLI={dev_accs[best_i_grid][1]:.2f}"})
        my_logger("Ext LR = %s"%('\t'.join(['%1.1e'%l for l in args.ext_learning_rate_vec])), color='green')
        my_logger("Dev Accs = %s"%dev_accs_s, color='green')
        my_logger("+++++++++++++", color='green')
        my_logger("BEST Ext LR: %1.0e, Acc=%1.2f%%" % (best_ext_lr, dev_accs[best_i_grid][0]), color='green')

        ## Loading best ext_emb model
        acc_s = ("%1.2f" % (dev_accs[best_i_grid][0]*100)).replace('.', '_')
        model_to_load = 'exp_%d_%s_ext_lr_grid_best_%d_loop_%d_%s.bin' % (wan.exp_id, server_name, best_i_grid, loop_counter, acc_s)
        my_logger("Loading best model of i= %d (should be: acc=%1.2f%%)  		filename: %s"%(best_i_grid, dev_accs[best_i_grid][0]*100, model_to_load), color='green')
        model.load_state_dict(torch.load(os.path.join(args.temp_models_dir, model_to_load)), strict=False)
        # TODO -remove next line (sanity check - making sure this is the correct model loaded):
        final_dev_acc, _, acc_by_label_str, dist_by_label_str = eval_examples_batch(dev_examples, model, data_type='dev')
        my_logger("Final Dev Acc (after loading): %1.2f%% %s\n\n"%(final_dev_acc*100, acc_by_label_str), color='green')
        my_logger("Final Dev Acc (after loading) by label:%s\n\n"%(dist_by_label_str), color='green')
        if final_dev_acc!=best_acc:
            my_logger("\n\n!!!!!! BUG: Best dev acc: %1.2f%%, acc of loaded model: %1.2f%%\n\n"%(best_acc*100, final_dev_acc*100), color='green')

        # MNLI Acc
        my_logger("----------------------- Evaluating MNLI set after finished grid search on ext_emb: -----------------------", color='green')
        mnli_acc, eval_after, eval_examples, eval_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)

        # logging final resutls @@@@@@@@@@@@@@@@
        my_logger("\n--- Finetuning Ext-Emb on Ext-LR, Results: ---\n BEST Ext-LR: %1.0e\n\tDev Acc: %1.2f%% \t\t%s\n" % (best_ext_lr, final_dev_acc * 100, acc_by_label_str) + '\tMNLI acc after ext_emb: %1.2f%% \t\t%s\n\t%s\n\t%s' % (mnli_acc * 100, eval_acc_by_label_str, mnli_dist_by_label_str, mnli_dist_by_label_str), highlight=('@', 84), color='green')

        # Saving best model
        model_to_save = 'exp_%d_ext_emb_finetuned.bin' % (wan.exp_id)
        my_logger('Saving best model to: %s' % model_to_save, color='green')
        torch.save(model.state_dict(), os.path.join(args.temp_models_dir, model_to_save))

        return model, best_acc, best_ext_lr, model_to_save


    def finetune_bert_only(model, model_to_load):
        """
        Fine-tune only BERT's parameters, while freezing S-KAR's.
        Scanning args.learning_rate_vec to find the best learning rate. by default, args.learning_rate_vec = [2e-5] without scanning
        """

        dev_accs = []
        for i_grid, lr in enumerate(args.learning_rate_vec):
            args.learning_rate = lr
            if len(args.learning_rate_vec) > 1:
                my_logger('\n\n==================  Grid Search LR=%1.1e (%d/%d) ===============\n' % (lr, i_grid, len(args.learning_rate_vec)-1), color='yellow')
            state_string = 'for Learning Rate %1.1e (%d/%d) (finetuning bert only)' % (lr, i_grid, len(args.learning_rate_vec))

            ### Loading model after finedtuning ext_emb
            if not args.finetune_vanilla:       # for finetuning vanilla BERT or RoBERTa models (without loading anything into them before the finetuning).
                my_logger("***Loading best ext_emb finetune model: %s" % model_to_load, color='yellow')
                model.load_state_dict(torch.load(os.path.join(args.temp_models_dir, model_to_load)), strict=False)
            else:   # Finetuning Vanilla:
                model = manage_init_and_load_model(args)  # Initializing and loading model according to arg.pretrained_model_to_load (=bert_mnli or roberta_mnli usually)


            ### Training bert modules
            optimizer = prepare_optimizer(model, lr)
            model, _, _, _, _ = train_model(model, train_examples, optimizer, lr, freeze_type='ext_emb_and_kar', nb_epochs=args.nb_vanilla_epochs)  # train model

            ### Evaluating model on Dev set
            my_logger('\nEvaluating Dev set accuracy %s):\n' % state_string, color='yellow')
            dev_acc, _, eval_acc_by_label_str, dist_by_label_str = eval_examples_batch(dev_examples, model, data_type='dev')
            my_logger('-------\nDev set accuracy %s=%1.2f%% %s\n%s\n' % (state_string, dev_acc*100, eval_acc_by_label_str, dist_by_label_str), color='yellow')

            ### MNLI Acc
            my_logger("----------------------- Evaluating MNLI set %s: -----------------------"%state_string, color='yellow')
            mnli_acc, _, _, mnli_eval_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)
            my_logger(
                "\n--- Dev Accuracy for %1.1e: ---\n \tAcc Dev Result: %1.2f%% \t\t\t\t\t%s\n" % (lr, dev_acc*100, mnli_eval_acc_by_label_str) + '\tMNLI acc after ext_emb: %1.2f%% \t\t%s\n\t%s\n\t%s' % (mnli_acc * 100, mnli_eval_acc_by_label_str, mnli_dist_by_label_str, mnli_dist_by_label_str),
                highlight=('@', 84), color='yellow')

            dev_accs.append((dev_acc, mnli_acc))
            dev_accs_s = ',\t'.join([' %1.2f/%1.2f' % (a1 * 100, a2 * 100) for (a1, a2) in dev_accs])   # dev acc / MNLI acc
            dev_accs_s_wan = ''.join(['[%1.0e: %1.1f/%1.1f]' % (l, a1 * 100, a2 * 100) for l, (a1, a2) in zip(args.learning_rate_vec, dev_accs)])   # dev acc / MNLI acc
            my_logger("For lr %1.1e\n \tAcc dev: %1.2f%%\n" % (lr, dev_acc*100) + '\n\tMNLI acc after ext_emb: %1.2f%%'%(mnli_acc*100), highlight=('=@', 20), color='yellow')

            best_i_grid = np.array([x[0] for x in dev_accs]).argmax()

            if args.save_every_grid_step:        # instead of saving only when the current model is superior to those of all previous inits (as implemented below)- here we save the model EVERY init
                save_model_finetuning(model, i_grid, 'lr_grid', dev_acc, mnli_acc)

            if best_i_grid == i_grid:
                best_acc = dev_acc
                save_model_finetuning(model, i_grid, 'lr_grid', best_acc, mnli_acc)

            wan.update({'Grid LR %d acc' % (i_grid): '%1.2f%%/%1.2f%%' % (dev_acc*100, mnli_acc*100)},
                       status='lr_i %d/%d, best acc=%1.2f%% (i_grid %d)' % (i_grid, len(args.learning_rate_vec), best_acc * 100, best_i_grid))


            my_logger('so far, best acc=%1.1f%% for i_grid=%d, ' % (best_acc * 100, best_i_grid), color='yellow')
            wan.update({'LR Grid Accs': dev_accs_s_wan})

        my_logger('='*60 + f'Finished Grid Searches for {len(args.learning_rate_vec)} Learning Rates' + '='*60, color='green')

        best_i_grid = np.array([x[0] for x in dev_accs]).argmax()
        best_lr = args.learning_rate_vec[best_i_grid]
        wan.update({'Best LR': f"{best_lr:.0e}| Acc={dev_accs[best_i_grid][0]:.2f}, MNLI={dev_accs[best_i_grid][1]:.2f}"})
        my_logger("LR = %s"%('\t'.join(['%1.1e'%l for l in args.learning_rate_vec])), color='cyan')
        my_logger("Dev Accs = %s"%dev_accs_s, color='cyan')
        acc_s = ("%1.2f" % (dev_accs[best_i_grid][0]*100)).replace('.', '_')
        model_to_load = 'exp_%d_%s_lr_grid_best_%d_loop_%d_%s.bin' % (wan.exp_id, server_name, best_i_grid, loop_counter, acc_s)
        my_logger("Loading best model of i= %d (should be: acc=%1.2f%%)  		filename: %s"%(best_i_grid, dev_accs[best_i_grid][0]*100, model_to_load), color='cyan')
        model.load_state_dict(torch.load(os.path.join(args.temp_models_dir, model_to_load)), strict=False)
        # TODO -remove next line (sanity check - making sure this is the correct model loaded):
        final_dev_acc, _, acc_by_label_str, dist_by_label_str = eval_examples_batch(dev_examples, model, data_type='dev')
        my_logger("Final Dev Acc (after loading): %1.2f%% %s\n\n"%(final_dev_acc*100, acc_by_label_str), color='cyan')
        my_logger("Final Dev Acc (after loading) by label:%s\n\n"%(dist_by_label_str), color='cyan')
        if final_dev_acc!=best_acc:
            my_logger("\n\n!!!!!! BUG: Best dev acc: %1.2f%%, acc of loaded model: %1.2f%%\n\n"%(best_acc*100, final_dev_acc*100), color='cyan')

        # MNLI Acc
        my_logger("----------------------- Evaluating MNLI set after finished grid search on ext_emb: -----------------------", color='cyan')
        mnli_acc, eval_after, eval_examples, eval_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)

        # logging final resutls @@@@@@@@@@@@@@@@
        my_logger("\n--- Finetuning BERT on LR, Results: ---\n BEST Ext-LR: %1.0e\n\tDev Acc: %1.2f%% \t\t\t%s\n" % (args.learning_rate_vec[best_i_grid], final_dev_acc * 100, acc_by_label_str) + '\tMNLI acc after ext_emb: %1.2f%% \t\t%s\n\t%s\n\t%s' % (mnli_acc * 100, eval_acc_by_label_str, mnli_dist_by_label_str, mnli_dist_by_label_str), highlight=('@', 84), color='cyan')

        # Saving best model
        model_to_save = 'exp_%d_finetuned.bin' % (wan.exp_id)
        my_logger('Saving best model to: %s' % model_to_save, color='cyan')
        torch.save(model.state_dict(), os.path.join(args.temp_models_dir, model_to_save))

        return model, best_acc, best_lr, model_to_save



    """=========================================================== MAIN ==========================================================="""
    """============================================================================================================================"""


    global item2class, class2ind
    item2class, class2ind = get_item2class()
    start_tic = ut.Tic()

    if args.train_group:    ### train_group is only relevant for training multiple training set at once. Have not been tested on this version yet.
        if args.train_group[0][:5]=='Loop_':       # for case of cross validation: --train_group Loop_numbers_task2_train
            assert  args.test_group[0][:5]=='Loop_'  # both train and test must be either cross_validation or not.
            train_group_list = [args.train_group[0][5:] + '[%d]'%i for i in range(5)]
            test_group_list = [args.test_group[0][5:] + '[%d]'%i for i in range(5)]
        else:
            train_group_list = ['+'.join(args.train_group)]
            test_group_list = ['+'.join(args.test_group)]
        nb_of_outer_loops = len(train_group_list)
    else:
        nb_of_outer_loops = 1
        train_group_list = [None]
        test_group_list = [None]

    for outer_loop in range(nb_of_outer_loops):
        # Each outer_loop is a one row in WanDB, but has multiple rows in the extraxt_results. Used when we want to train on multiple training sets. haven't been tested on this version

        ## Setting up wandb
        start_runtime = ut.now_str()
        start_runtime_s = start_runtime[5:19].replace(':', '-').replace(' ', '_')  # for filenames
        logger_dir = 'logger'
        wan = gu.WandbExperiment(wandb_project_name, args, start_runtime, server_name, repository_dir_and_files, wandb_off=args.wandb_off)   # Initialize a google sheet row (before adding it)
        data = gu.sys_args_from_args(args, sys.argv)
        wan.update(data, status='Getting ready...')

        # Defining loggers
        global logger, logger_debug
        formatter = logging.Formatter('%(asctime)s %(message)s')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        log_filename = os.path.join(logger_dir, 'exp_%d_%s_server_%s.log' % (wan.exp_id, start_runtime_s, server_name))
        log__debug_filename = os.path.join(logger_dir, 'exp_%d_%s_server_%s_debug.log' % (wan.exp_id, start_runtime_s, server_name))
        logger = setup_logger(formatter, 'first_logger', log_filename)
        logger_debug = setup_logger(formatter,'second_logger', log__debug_filename)

        train_group = train_group_list[outer_loop]
        test_group = test_group_list[outer_loop]

        if train_group:
            args.train_setname = group_to_set(train_group).strip().split(' ')
            assert not any([s=='' for s in args.train_setname])
        if test_group:
            args.test_setname = group_to_set(test_group).strip().split(' ')
            assert not any([s == '' for s in args.test_setname])
        train_setname_str = ', '.join(args.train_setname)
        dev_setname_str = ', '.join(args.dev_setname)
        test_setname_str = ', '.join(args.test_setname)


        lr_list, max_train_examples_list, max_mix_MNLI_examples_list, seed_list, loop_param, loop_val_list = get_loop_params()  # gets the parameters for each iteration of the outer_loop


        init_seed(seed_list[0])
        init_torch_seed(seed_list[0])
        loop_results = []
        loop_best_acc_results = []
        loop_MNLI_results = []
        loop_param, loop_val_list, arg2, vals2 = get_iterable_args(args, excluded_loop_args)

        # This loop is for grid search over parameters set in get_iterable_args (e.g. lr)
        for loop_counter, loop_val in enumerate(loop_val_list, 1):
            setattr(args, loop_param, loop_val)
            lr, max_train_examples, max_mix_MNLI_examples, seed = backward_compatability_loop_variables(args)
            nb_loops = len(loop_val_list)
            args.lr = lr
            if '--lr' not in sys.argv:
                sys.argv += ['--lr', str(lr)]
            if not args.no_seed_in_loop:
                init_seed(seed)
                init_torch_seed(seed)
            start_time = ut.now_str()
            start_stamp = get_start_stamp()
            loop_val_s = ('%1.1e'%loop_val) if loop_param in ['learning_rate', 'ext_learning_rate'] else '%d'%loop_val
            my_logger('\n\n\n***************************************** Loop #%d: %s = %s **************************************'%(loop_counter, loop_param, loop_val_s), color='cyan')
            my_logger('\n' + start_stamp, color='cyan')

            my_logger('\n\n------------------  Step 1: Preparing Training and Test Sets ------------\n', color='cyan')
            test_examples, test_templates = prepare_test_examples()
            dev_examples, dev_templates = prepare_dev_examples()
            train_examples, train_templates = prepare_training_examples()

            # saving the local_wiki dictionary. local_wiki is cache for entity-attribute pairs extracted from Wikidata
            if not args.finetune_vanilla:
                if args.EP in ['Color']:
                    wnu.save_local_wiki_features(local_wiki_features)
                elif args.EP in ['Dir_Hypernymy', 'Location', 'Trademark']:
                    wnu.save_local_wiki(local_wiki)
                elif args.EP in ['Combined']:
                    wnu.save_local_wiki_features(local_wiki_features)
                    wnu.save_local_wiki(local_wiki)

            #### ONLY TEST - Loading existing model for just testing
            if args.inferbert_to_load:
                model = load_existing_inferbert_model(args.inferbert_to_load)
                test_acc, mnli_accuracy = eval_dataset_and_mnli(model, 'bert', 'test')
                results_only_test_to_wan(test_acc, mnli_accuracy)
                my_logger('run time = %1.0f secs' % start_tic.toc(is_print=False), color='cyan')
                exit(0)

            ### Finetune Ext Weights Or Load a model
            if not args.load_ext_emb_model and not args.finetune_vanilla:     # Finetune Ext-Emb Weights as usual
                ### Finetune Ext Weights - fune-tuning the S-KAR waits while freezing the rest of the model
                my_logger('\n\n ********   FineTuning Ext Weights   ***********\n***********************************************\n***********************************************', color='green')
                model, best_ext_acc, best_ext_lr, model_filename = fineture_ext_weights()
            else:           # Skipping Ext-Emb finetuning by Loading already finetuned ext_emb model instead
                my_logger('\n\n ********  !! Skipping to loading existing Ext-Emb finetuned model %s ***********\n***********************************************\n***********************************************' % args.load_ext_emb_model, color='green')
                # Model will be loaded from models/taught_models/temp_models. For example: python train_two_steps.py  --load_ext_emb_model exp_2747_nlp06_ext_lr_grid_best_1_loop_1_76_52.bin
                model = manage_init_and_load_model(args)  # Initializing and loading model according to arg.pretrained_model_to_load (=bert_mnli or roberta_mnli usually)
                model_filename = args.load_ext_emb_model    # This model will be loaded in finetune_bert_only()
                best_ext_acc, best_ext_lr = -1, -1

            ### Finetune BERT
            my_logger('\n\n ******************************************  %s  ******************************************\n**********************************************************************************************************************************************\n' % ('Training BERT only (freezing ext_emb (and KAR))'), color='cyan')
            model, best_dev_acc, best_lr, finetuned_model_name = finetune_bert_only(model, model_filename)

            ### Evaluating model on the test set and on MNLI dev set
            my_logger('\n\n\n\n\n ******************************************  %s  ******************************************\n**********************************************************************************************************************************************\n' % ('Testing Model for Loop %d'%loop_counter), color='red')
            test_acc, mnli_accuracy = eval_dataset_and_mnli(model, 'bert', 'test')

            save_taught_model(model)

            ### Evaluating (again) on MNLI dev set - there is a redundency with evaluating MNLI twice. to be fixed.
            my_logger('\n\n------------------  Step 5: Evaluating MNLI-Dev and Teaching-Test sets AFTER Teaching  ------------\n(%s)'%(ut.now_str()), color='cyan')
            mnli_acc, eval_after, mnli_eval_examples, mnli_acc_by_label_str, mnli_dist_by_label_str = get_MNLI_dev_acc(args.data_dir, model)
            result, eval_loss, eval_acc_by_label_str, dist_by_label_str = eval_examples_batch(test_examples, model, data_type='test')
            my_logger("\n--- Final Loop %d Results: ---\n \tAcc Test Result (after loading): %1.2f%% %s\n" % (loop_counter, result * 100, eval_acc_by_label_str) + '\tMNLI acc after ext_emb: %1.2f%% %s\n\t%s\n\t%s' % (mnli_acc * 100, mnli_acc_by_label_str, dist_by_label_str, mnli_dist_by_label_str), highlight=('@',84), color='cyan')

            loop_results.append(result)
            loop_best_acc_results.append(best_dev_acc)
            loop_MNLI_results.append(mnli_acc)
            # results_to_googlesheet(loop_results, loop_MNLI_results, loop_best_acc_results, best_dev_acc, best_ext_lr, best_lr, finetuned_model_name)
            results_to_wan(loop_results, loop_MNLI_results, loop_best_acc_results, test_acc, best_ext_acc, best_dev_acc, best_ext_lr, best_lr, finetuned_model_name, mnli_accuracy)

    # gse.update(status='Done')
    wan.update(status='Done')
    # wandb.log({'Status': 'Done'})
    my_logger('run time = %1.0f secs'%start_tic.toc(is_print=False), color='cyan')

if __name__ == "__main__":
    main()
