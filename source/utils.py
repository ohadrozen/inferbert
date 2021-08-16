import numpy as np
import pickle
import datetime
import time
import re
import os
import random
import pandas as pd
import torch
import json
import csv

class Tic(object):
    def __init__(self):
        self.t0 = time.time()

    def toc(self, is_print=True):
        if is_print:
            print('%1.3f'%(time.time() - self.t0))
        return time.time() - self.t0

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    if axis==0:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x,1).reshape(-1,1))
        return e_x / np.sum(e_x,1).reshape(-1,1)

def load_pickle(filename, var_type =''):
    if var_type=='pandas':
        output = pd.read_pickle(filename)
    else:
        with open(filename, 'rb') as handle:
            output = pickle.load(handle)
    return output

def save_pickle(filename, input):
    if isinstance(input, pd.DataFrame):     # dealing with Pandas separately
        pd.to_pickle(input, filename)
    else:
        MAXATTEMPTS = 15
        for attempt in range(MAXATTEMPTS):
            with open(filename , 'wb') as handle:
                pickle.dump(input, handle)
            # checking if file was indeed saved
            if os.path.isfile(filename):
                break
            else:
                print("Warning!!!!!! File %s wasn't created. Trying again (attempt %d/%d)"%(filename, attempt, MAXATTEMPTS))

def save_json(filename, dict_in):
    with open(filename, 'w') as fp:
        json.dump(dict_in, fp, indent=4)

def load_json(filename):
    with open(filename,'r') as f:
        dict_out = json.load(f)
    return dict_out


def read_csv(filename, max_length=100000000, cols=None, delimiter=',', do_print=True):
    """ designed to read one of the output files in '../data/cps/pipeline_step' into a list of dicts """
    with open(filename, mode='r', encoding='utf8') as infile:
        reader = csv.reader(infile, delimiter=delimiter)
        examples = []
        i = 0
        if do_print:
            for row in reader:
                if i == 0:
                    if cols is None:
                        cols = row
                    i += 1
                    continue
                if i > max_length:
                    break
                d = {cols[j]: row[j] for j in range(min(len(cols), len(row)))}
                for eval_column in ['knn', 'similar_questions']:
                    if eval_column in d:
                        d[eval_column] = eval(d[eval_column])
                examples.append(d)
                i += 1
        else:
            for row in reader:
                if i == 0:
                    if cols is None:
                        cols = row
                    i += 1
                    continue
                if i > max_length:
                    break
                d = {cols[j]: row[j] for j in range(min(len(cols), len(row)))}
                for eval_column in ['knn', 'similar_questions']:
                    if eval_column in d:
                        d[eval_column] = eval(d[eval_column])
                examples.append(d)
                i += 1
    if do_print:
        print(f'{len(examples)} rows were read from {filename}')
    return examples

def now_str():
    return str(datetime.datetime.now())[:-7]

def d_inc(d, key, val=1):
    if key in d.keys():
        d[key] += val
    else:
        d[key] = val


def dump_dict_to_csv(filename, dict_list):
    """ dumps a list of dicts to csv. all columns(=keys) must appear on the first item in the list """
    if dict_list == []:
        print("list is empty. Couldn't save anything to csv")
        return

    keys = list(dict_list[0].keys())
    print('Writing %d rows to %s' % (len(dict_list), filename))
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    print('Writing completed')


def dump_dict_to_tsv(filename, dict_list):
    """ dumps a list of dicts to csv. all columns(=keys) must appear on the first item in the list """
    if dict_list == []:
        print("list is empty. Couldn't save anything to tsv")
        return

    keys = list(dict_list[0].keys())
    print('Writing %d rows to %s' % (len(dict_list), filename))
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter='\t')
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    print('Writing completed')


def dump_list(filename, str_list):
    # dumps a list of strings to text file
    with open(filename,'w') as f:
        for s in str_list:
            f.write(s + '\n')

def find_after(s, target, begin, end = None):
    loc = s[begin:end].find(target)
    if loc>-1:
        return loc + begin
    else:
        return loc

def find_str_after_str(s, str_before, arg_type = 'single_word', end_chars = '$', ignore_case = False):
    """
    Looks for an string after the specified substring. For example: find_str_after_str("Verb: walk (bla bla bla) (ish ish)", 'Verb:') will return 'walk'
    :param s: full string
    :param str_before: the sub-string we want to look for and then exclude from the result
    :param arg_type: 'single_word' - only one word. will ignore one spaces befoer it. only takes alphabet word. not digits.
                     'single_alphanumeric' - single alphanumeric word (including underscores)
                     'multiple_words' - all words including spaces from the set_before to the end of the string/end_chars
                     'all_between' - returns all expressions bounded between str_before and end_chars. so for find_str_after_str("Verb: walk (bla bla bla) (ish ish)", 'all_between', '\(', '\)')
                                    it will return ['bla bla bla', 'ish ish'].
    :param end_chars: a string contains all characters to end the search (only relevant in case of more than one word).
                    For example, find_str_after_str("Verb: walk talk; (bla bla bla)", 'Verb:', 'multiple_words',';') will return 'walk talk'
                    Please note! use backslash ('\') for special characters. E.g. '\*' instead of '*'
    :param cased: case sensitive or not (True = case sensitive)
    :return: the wanted argument
    """
    re_cased = re.IGNORECASE if ignore_case else re.ASCII

    if arg_type == 'single_word':
        re_str = r"(?<=%s)\s*[a-zA-Z]+"%(str_before)
    elif arg_type == 'single_alphanumeric':
        re_str = r'(?<=%s)\s*\w+'%(str_before)
    elif arg_type == 'multiple_words':
        re_str_with_end = r'(?<=%s).*?(?=[%s])'%(str_before, end_chars)
        re_str_without_end = r'(?<=%s).*'%(str_before)
        re_res_with_end = re.search(re_str_with_end, s, re_cased)
        re_res_without_end = re.search(re_str_without_end, s, re_cased)
        re_str = re_str_with_end if re_res_with_end else re_str_without_end
    elif arg_type == 'all_between':
        re_str = r'(?<=%s).*?(?=[%s])'%(str_before, end_chars)
        re_res = re.findall(re_str, s, re_cased)
        return re_res
    else:
        raise ValueError ("wrong arg_type")

    re_res = re.search(re_str, s, re_cased)
    arg = re_res.group().strip().lower() if re_res else ''
    return arg


def str2num(s_list, output_type = 'float'):
    if type(s_list)==str:
        try:
            if output_type == 'float':
                return float(s_list)
            if output_type == 'int':
                return int(s_list)
            if output_type == 'percent':
                return float(s_list.strip('%')) / 100.0
            if output_type == 'percent*100':
                return float(s_list.strip('%'))
        except:
            return ['---'] * len(s_list)
    else:
        try:
            if output_type == 'float':
                return [float(s) for s in s_list]
            if output_type == 'int':
                return [int(s) for s in s_list]
            if output_type == 'percent':
                return [float(s.strip('%')) / 100.0 for s in s_list]
            if output_type == 'percent*100':
                return [float(s.strip('%')) for s in s_list]
        except:
            return ['---']*len(s_list)


def semiRandomSample(input, k, indices_path='random_indices'):
    # returning a deterministic pseudo-ramdon l indices for a give max_length, so the order doesn't depend on any seed (saved on disk)
    # Example:
    # for max_length=5 we can have:
    #       semiRandom_ind(1,5) = [3]
    #       semiRandom_ind(3,5) = [3, 2, 4]
    #       semiRandom_ind(5,5) = [3, 2, 4, 5, 1]

    assert k <= len(input)

    max_length = len(input)

    if not os.path.exists(indices_path):
        os.makedirs(indices_path)
    indices_filename = os.path.join(indices_path, 'indices_len_%d.pkl' % max_length)

    if os.path.isfile(indices_filename):
        indices = load_pickle(indices_filename)
    else:
        indices = list(range(max_length))
        random.shuffle(indices)
        save_pickle(indices_filename, indices)

    ind = indices[:k]
    output = [input[i] for i in ind]
    return output

def is_number(s):
    # returns if the string represents a number or not.
    # Number includes 0-9, '.', or '-'. so "-0.41" is a number.
    s = s.strip()
    chars_ok = not re.search(r'[^0-9\.\-]',s)
    minus_ok = s.find('-') < 1 and len(re.findall(r'\-',s)) <=1
    dot_ok = len(re.findall(r'\.',s)) <=1

    return chars_ok and minus_ok and dot_ok

def pdf(df):
    print(df.to_string())

def freeze_weights(inner_model_path, models=None, defreeze=False):
    # input:
    #   inner_model_path: the model to freeze, OR the model path, e.g. model.bert.attention
    #   models: list of models to freeze within inner_model_path

    def freeze_single_model_weights(inner_model, defreeze):
        if not defreeze:
            for p in inner_model.parameters():
                p.requires_grad = False
        else:
            for p in inner_model.parameters():
                p.requires_grad = True

    if models == None:  # inner_model_path is not a path, but the name of the model
        inner_model = inner_model_path
        freeze_single_model_weights(inner_model, defreeze)
    else:   # path and then a list of models within this model
        for model_s in models:
            full_modelname = eval('inner_model_path.%s'%model_s)
            freeze_single_model_weights(full_modelname, defreeze)

def zero_weights(inner_model_path, models=None, val=0):
    # input:
    #   inner_model_path: the model to freeze, OR the model path, e.g. model.bert.attention
    #   models: list of models to freeze within inner_model_path

    def zero_single_model_weights(inner_model, val):
        for p in inner_model.parameters():
            p.data.fill_(val)

    if models == None:  # inner_model_path is not a path, but the name of the model
        inner_model = inner_model_path
        zero_single_model_weights(inner_model, val)
    else:   # path and then a list of models within this model
        for model_s in models:
            full_modelname = eval('inner_model_path.%s'%model_s)
            zero_single_model_weights(full_modelname, val)

def random_weights(inner_model_path, models=None, var=1):
    # input:
    #   inner_model_path: the model to freeze, OR the model path, e.g. model.bert.attention
    #   models: list of models to freeze within inner_model_path

    def random_single_model_weights(inner_model, var):
        for p in inner_model.parameters():
            p.data.uniform_(0.0, var)


    if models == None:  # inner_model_path is not a path, but the name of the model
        inner_model = inner_model_path
        random_single_model_weights(inner_model, var)
    else:   # path and then a list of models within this model
        for model_s in models:
            full_modelname = eval('inner_model_path.%s'%model_s)
            random_single_model_weights(full_modelname, var)

def random_weights_keep_var(inner_model_path, models=None):
    # input:
    #   inner_model_path: the model to freeze, OR the model path, e.g. model.bert.attention
    #   models: list of models to freeze within inner_model_path

    def zero_single_model_weights(inner_model):
        for p in inner_model.parameters():
            t = p.data.cpu().numpy()
            np.random.shuffle(t)
            p.data = torch.from_numpy(t)


    if models == None:  # inner_model_path is not a path, but the name of the model
        inner_model = inner_model_path
        zero_single_model_weights(inner_model)
    else:   # path and then a list of models within this model
        for model_s in models:
            full_modelname = eval('inner_model_path.%s'%model_s)
            zero_single_model_weights(full_modelname)

if __name__ == '__main__':
    # s ='# blan Verb: verb'
    # loc = find_after(s, '*', 1)
    # print(loc)
    # print(s[loc+1:])

    # print(find_str_after_str("'*fsaaaaaa* *bbb* *cccc*'", '\*', 'all_between', '\*'))
    # print(find_str_after_str('*fsaaaaaa* *bbb* *cccc*', '\*fs'))

    # find_str_after_str('# Verb:   th4is is the; verb', 'Verb:', 'multiple_words')
    # print(    find_str_after_str(s, 'Verb:', 'multiple_words'))

    # print(str2num(['0%','1%','2.4%'],'percent'))

    # print(semiRandomSample([10,11,12,13,14,15,16,17],2))

    print(is_number('-123.4'))
    print(is_number('123.4'))
    print(is_number('1234'))
    print(is_number('.1234'))
    print(is_number('1234.'))
    print(is_number('123.4.'))
    print(is_number('12-3.4'))
    print(is_number('-123.4-'))
    print(is_number('.123.4'))