import nltk
import json
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np
import utils
import spacy
import inflect  # for getting plural
import pandas as pd
import warnings
import time
# from utils import tic
import wikidata_queries as wiki
from SPARQLWrapper import SPARQLWrapper, JSON
from termcolor import colored
from country_to_adjective import country2adj
import re

#
print("** Loading inflect.engine()...")
p = inflect.engine()
#
print("**Loading Spacy...")
nlp = spacy.load('en_core_web_lg')
print("Done loading Spacy")
pos_dict = utils.load_pickle('wordnet_pos.p')  # word and it's possible POS tags
lemmatizer = WordNetLemmatizer()
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

global excluded_classes, singular_errors
excluded_classes = ['object', 'thing', 'attribute', 'abstraction', 'entity', 'object', 'whole', 'artifact', 'physical_entity', 'substance', 'psychological_feature', 'matter', 'cognitive_state', 'causal_agent', 'living_thing', 'event', 'act',
                    'being', 'state']
singular_errors = []

#
# syn = wn.synsets("flower")[0]
# hypernym_syns = syn.hypernyms() + syn.instance_hypernyms()
# hypernyms = set([lemma for hyp_syn in hypernym_syns for lemma in hyp_syn.lemma_names()])

def all_hypernyms(word, sense=0, is_print=True):
    results = []
    if wn.synsets(word):
        hypernyms = wn.synsets(word)[sense]
        while hypernyms.hypernyms():
            hypernyms = hypernyms.hypernyms()[0]
            # print(hypernyms).lemmas()
            results.append(hypernyms)
            if is_print:
                print(' ; '.join([l.name() for l in hypernyms.lemmas()]))
        return results

def all_hypernyms2(word, noun_only=False):
    results = []
    if wn.synsets(word):
        for sense in wn.synsets(word):
            if not noun_only or sense.pos()=='n':
                for path in sense.hypernym_paths():
                    results += path[:-1]
        return list(set(results))
    return []

def all_hypernyms_names(word, keep_plurality=True, MaxHypernymys=9):
    # returns the full hypernyms path for 'word'. Used for turkers (filters '_' and high level hypernym classes)
    # if keep_plurality==True, converts all results to singular or plural according to 'word'
    global excluded_classes
    MinHyponyms = 12
    results = []
    if wn.synsets(word):
        senses = wn.synsets(word)
        # hypernyms = wn.synsets(word)[sense]
        for hypernyms in senses:
            while hypernyms.hypernyms():
                hypernyms = hypernyms.hypernyms()[0]
                # print(hypernyms).lemmas()
                lemma_name = hypernyms.lemmas()[0].name()
                if lemma_name.find('_') < 0 and lemma_name not in excluded_classes and len(results)< MaxHypernymys:
                    # # making sure each hypernym has at least MinHyponyms(=12) hyponyms
                    # hyponyms = all_hyponyms_by_freq(lemma_name, word)
                    # if hyponyms and len(hyponyms) >= MinHyponyms:
                    results.append(lemma_name)
                # print(' ; '.join([l.name() for l in hypernyms.lemmas()]))
        results = list(set(results))
        return change_plurality(word, results) if keep_plurality else results
#
# def find_hypernym_sense_to_hyponym(hyper, example_hypo):
#     example_hypo_l = lemmatizer.lemmatize(example_hypo.strip())
#     hyper_l = lemmatizer.lemmatize(hyper.strip())
#     hyper_sense = find_relevant_sense(wn.synsets(hyper_l), example_hypo_l)
#     if hyper_sense is None: # trying different combinations of no lemmatizing
#         hyper_sense = find_relevant_sense(wn.synsets(hyper), example_hypo_l)
#         if hyper_sense is None:
#             hyper_sense = find_relevant_sense(wn.synsets(hyper_l), example_hypo)
#             if hyper_sense is None:
#                 hyper_sense = find_relevant_sense(wn.synsets(hyper), example_hypo)
#     return hyper_sense
#

def find_relevant_sense(hyper_synsets, target_hyponym):
    # from a few given senses (synsets), find the one with the target_hyponym
    for s in hyper_synsets:
        if target_hyponym in to_singular([l.name() for l in s.lemmas()]):
            return s
        found = find_relevant_sense(s.hyponyms(), target_hyponym)
        if found:
            return s
    return None

def all_hyponyms(word, example_hyponym):

    def all_hyponyms_rec(synsets, level=0):
        for s in synsets:
            if any([l.name() == example_hyponym for l in s.lemmas()]):
                continue
            indent = '\t' * level
            if '-' not in s.name() and '_' not in s.name():
                print(indent + ' / '.join([l.name() for l in s.lemmas()]))
            h = s.hyponyms()
            all_hyponyms_rec(h, level+1)

    example_hyponym_lematized = lemmatizer.lemmatize(example_hyponym)
    sense = find_relevant_sense(wn.synsets(word), example_hyponym_lematized)
    if sense is None:   # sometimes lemmatizer makes mistakes (e.g. lemmatizer('boss')='bos')
        sense = find_relevant_sense(wn.synsets(word), example_hyponym)
        if sense is None:
            raise ValueError("sense = None")
    all_hyponyms_rec([sense])

def all_hyponyms_by_freq(word, example_hyponym, max_results=50, keep_plurality=True, low_freq=False):

    def all_hyponyms_rec(synsets, level=0):
        for s in synsets:
            if any([l.name() == example_hyponym for l in s.lemmas()]):
                continue
            if '-' not in s.name() and '_' not in s.name():
                for l in s.lemmas():
                    if l.name() != word and l.name() not in sense_lemmas and l.name().find('_') < 0:
                        results.append(l.name())
                        freq.append(l.count())
            h = s.hyponyms()
            # if level>6:
            #     print("Warning: for word=%s and example_hyponym=%s level>6. skipping"%(word, example_hyponym))
            all_hyponyms_rec(h, level+1)


    results = []
    freq = []
    original_word = word
    word, example_hyponym = to_singular([word, example_hyponym])
    example_hyponym_lematized = lemmatizer.lemmatize(example_hyponym)
    sense = find_relevant_sense(wn.synsets(word), example_hyponym_lematized)
    if sense is None:   # sometimes lemmatizer makes mistakes (e.g. lemmatizer('boss')='bos')
        sense = find_relevant_sense(wn.synsets(word), example_hyponym)
        if sense is None:
            # print("Error: sense = None")
            # raise ValueError("sense = None")
            return None
    sense_lemmas = [l.name() for l in sense.lemmas()]
    all_hyponyms_rec([sense])

    ind = np.array(freq).argsort()[::-1] if not low_freq else np.array(freq).argsort()[::1]
    s_results = [results[i] for i in ind][:max_results]
    return change_plurality(original_word, s_results) if keep_plurality else s_results

def is_cohyponym_for_similar_words(word, hyper_sense, start_time):
    # checks if the input 'word' is a co-hyponym of the 'hypo' with a common 'hyper'. Assuming singular input: hypo and hyper

    def find_hyponyms(synsets, target_hyponym):
        # from a few given senses (synsets), find if the target_hyponym exists. Assuming singular inputs
        for s in synsets:
            if target_hyponym in to_singular([l.name() for l in s.lemmas()]):
                print('found %s in %s'%(target_hyponym, s))
                return True
            found = find_hyponyms(s.hyponyms(), target_hyponym)
            if found:
                return found
        return False

    TimeOut = 3 #seconds
    if hyper_sense is None:     # if we get None as an input for hyper_sense (means there was a bug somehow finding the relevant sense) - return it's not a co-hyponym
        return False
    word_l = lemmatizer.lemmatize(word.strip())
    t1 = utils.Tic()
    found = find_hyponyms([hyper_sense], word_l)
    print("find_hyponyms took ",t1.toc())
    return found





    ### Old:
    TimeOut = 3 #seconds
    if hyper_sense is None:     # if we get None as an input for hyper_sense (means there was a bug somehow finding the relevant sense) - return it's not a co-hyponym
        return False
    word_l = lemmatizer.lemmatize(word.strip())
    t1 = utils.Tic()
    found = find_hyponyms([hyper_sense], word_l)
    print("find_hyponyms took ",t1.toc())
    return found


def sentence_hypernyms(sentence):
    tokens = word_tokenize(sentence)
    for w in tokens:
        h = all_hypernyms(w)
        if h:
            print('***' + w + ':')
            for hi in h:
                print(hi)
            print('\n')


def list_hypernyms(word_list, is_print=False):
    output = []
    for w in word_list:
        h = all_hypernyms(w, is_print=is_print)
        if h:
            output.append((w, h))
            if is_print:
                print('*** ' + w + ':')
                for hi in h:
                    print(hi)
                print('\n')
    return output

def list_hypernyms2(word_list, is_print=False, noun_only=False):
    global excluded_classes
    output = []
    for w in word_list:
        h = [hyper for hyper in all_hypernyms2(w, noun_only=noun_only) if synset_name(hyper) not in excluded_classes and synset_name(hyper).find('_')<0]
        if h:
            output.append((w, h))
            if is_print:
                print('*** ' + w + ':')
                for hi in h:
                    print(hi)
                print('\n')
    return output

def list_synonyms(word_list, is_print=False, noun_only=False):
    global excluded_classes
    output = []
    for w in word_list:
        h = [hyper for hyper in all_synonyms_antonyms(w)[0] if hyper.find('_') < 0]
        if h:
            output.append((w, h))
            if is_print:
                print('*** ' + w + ':')
                for hi in h:
                    print(hi)
                print('\n')
    return output

def pick_nouns_for_hypernyms(sentence, mode='noun'):
    doc = nlp(sentence)
    entities = [a.text for a in doc.ents]   # entities from SpiCy NER
    nouns = []
    pos_list = dict(include_propn=['NOUN', 'PROPN'], noun=['NOUN'])
    if mode == 'no_chunk':
        for d in doc:
            cand_is_upper = (d.text != sentence[0:len(d.text)] and d.text[0].isupper())
            if d.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not cand_is_upper:
                nouns.append(d.text.lower())
    else:
        for chunk in doc.noun_chunks:   # looking for NP chunks
            cand = chunk.root
            cand_is_upper = (cand.text != sentence[0:len(cand.text)] and cand.text[0].isupper())
            loc_after_chunk = (' ' + sentence + ' ').find(chunk.root.text) +len(chunk.root.text) -1
            loc_after_chunk = loc_after_chunk-1 if loc_after_chunk >= len(sentence) else loc_after_chunk    # In case the chunk ends where the sentence ends (so no character after it)
            has_dash = (sentence[(' ' + sentence + ' ').find(chunk.root.text) -2] == '-') or (sentence[loc_after_chunk] == '-')
            if cand.text not in entities and not cand_is_upper and cand.pos_ in pos_list[mode] and not has_dash:
                nouns.append(chunk.root.text.lower())
    return list(set(to_singular(nouns))), doc

def pick_nouns_for_color(sentence):
    doc = nlp(sentence)
    nouns = []
    for chunk in doc.noun_chunks:   # looking for NP chunks
        cand = chunk.root
        loc_after_chunk = (' ' + sentence + ' ').find(chunk.root.text) +len(chunk.root.text) -1
        loc_after_chunk = loc_after_chunk-1 if loc_after_chunk >= len(sentence) else loc_after_chunk    # In case the chunk ends where the sentence ends (so no character after it)
        has_dash = (sentence[(' ' + sentence + ' ').find(chunk.root.text) -2] == '-') or (sentence[loc_after_chunk] == '-')
        if cand.pos_ in ['NOUN', 'PROPN'] and not has_dash:
            nouns.append(chunk.root.text)
    return list(set(to_singular(nouns))), doc

def get_hypernym_candidates(sentence_string, is_print=True):
    nouns, doc = pick_nouns_for_hypernyms(sentence_string)
    words_synonyms = list_synonyms(nouns, is_print=is_print)
    return words_synonyms, doc

def get_hypernym_candidates2(sentence_string, is_print=True, mode='noun'):
    # look for the last word in each NP and returns all of its hypernyms, for all senses
    nouns, doc = pick_nouns_for_hypernyms(sentence_string, mode=mode)
    words_hypernyms = list_hypernyms2(nouns, is_print=is_print, noun_only=True)
    return words_hypernyms, doc, nouns

def get_synonym_candidates(sentence_string, is_print=True):
    # look for the last word in each NP and returns all of its hypernyms, for all senses
    nouns, doc = pick_nouns_for_hypernyms(sentence_string)
    words_hypernyms = list_hypernyms2(nouns, is_print=is_print, noun_only=True)
    return words_hypernyms, doc, nouns

def load_multinli(NLIFile, start_ind=0, length=100):
    sents = []
    with open(NLIFile, encoding='utf-8') as f:
        for i, jline in enumerate(tqdm(f.read().split('\n')[start_ind:start_ind+length],'Loading multinli')):
            result_t = json.loads(jline)
            p = result_t['sentence1']
            if i==65:
                pass
            h = result_t['sentence2']
            genre = result_t['genre']
            sents.append((p,h, genre))
    return sents


def load_multinli_w_label(NLIFile, start_ind=0, length=100):
    sents = []
    with open(NLIFile, encoding='utf-8') as f:
        for i, jline in enumerate(tqdm(f.read().split('\n')[start_ind:start_ind+length],'Loading multinli')):
            result_t = json.loads(jline)
            p = result_t['sentence1']
            if i==65:
                pass
            h = result_t['sentence2']
            genre = result_t['genre']
            label = result_t['gold_label']
            sents.append((p,h, genre, label))
    return sents

def multinli2hypernums(NLIFile, outputFile, start_ind=0, length=100):
    sents = load_multinli(NLIFile, start_ind, length)
    with open(outputFile + 'dobj.txt', 'w', encoding="utf-8") as fw:
        for s in tqdm(sents, desc="sentence"):
            fw.write('\n' + s + '\n')
            words_hypernyms, doc = get_hypernym_candidates(s)
            for (w, hs) in words_hypernyms:
                fw.write('*** ' + w + ':\n')
                for hi in hs:
                    fw.write(hi.name().split('.')[0] + ' -> ')
                fw.write('\n')

def synset_name(synset):
    if synset is None:
        return ''
    return synset.name().split('.')[0]

def search_multinli_hyper_pairs(NLIFile, outputFile, start_ind=0, length=1000):
    global excluded_classes
    sents = load_multinli_w_label(NLIFile, start_ind, length)
    with open(outputFile + 'dobj.txt', 'w', encoding="utf-8") as fw:
        for s in sents:
            p, h, _, l = s
            words_hypernyms, doc = get_hypernym_candidates2(h, is_print=False)
            for (w, hs) in words_hypernyms:
                # print('*** ' + w + ':')
                for hi in hs:
                    hyper_s = synset_name(hi)
                    # print(hyper_s + ' -> ')
                    if p.find(hyper_s)>-1 and h.find(hyper_s)<0 and p.find(w)<0:    # hyper in premise but not in hypothesis, and the word is not in the hypothesis.
                        print('\n' + p)
                        print(h)
                        print(l)
                        print('*** Found in H:%s -> %s'%(w, hyper_s))

def search_multinli_hyper_pairs_test(NLIFile, outputFile, start_ind=0, length=1000):
    global excluded_classes

    total = multiple_pairs = 0
    sents = load_multinli_w_label(NLIFile, start_ind, length)
    with open(outputFile + 'dobj.txt', 'w', encoding="utf-8") as fw:
        for i,s in enumerate(sents):
            if i%100==0: print('************',i)
            p, h, _, l = s
            pairs = find_hyper_hypo_pairs(p, h)
            if len(pairs)>1:
                multiple_pairs +=1
            if pairs:
                total  +=1
        print("\n\ntotal sentences: %d;  total sentences with pairs: %d; multiple pairs: %d; hypernymy/sent: %1.1f%%"%(len(sents), total, multiple_pairs, 100.*total/len(sents)))

def find_all_ind(val, val_list):
    return [i for i, x in enumerate(val_list) if x.lower() == val.lower()]

def find_all_country_ind(country, tokens):
    # example: country_orig='united states of america' ; tokens = ['i', 'used', 'to', 'live', 'in', 'the', 'u', '.', 's', '.', 'a', 'when', 'i', 'was', 'young']
    synonyms = {'united states of america': ['united states of america', 'the united states of america', 'the united states', 'united states', 'us', 'usa', 'america', 'the us', 'the usa', 'u . s .', 'u . s', 'the u . s .', 'the u . s . a .',
                                              'the u . s . a', 'u . s . a', 'u . s . a .'], 'united kingdom':['kingdom of england', 'england', 'the united kingdom', 'united kingdom', 'uk', 'u . k .', 'the u . k .', 'great britain', 'britain']}
    indices = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if country in synonyms:
            for c_syn in synonyms[country]:
                equal = True
                for j, ct in enumerate(c_syn.split()):
                    if i + j >= len(tokens) or tokens[i + j] != ct:
                        equal = False
                        break
                if equal:
                    indices.append(i)
                    i += len(c_syn.split())
                    break
        else:
            if country == t:
                indices.append(i)
        i += 1
    return indices

def find_all_ind2(target, tokens):
    """
    finds all appearances of a target phrase (could be multiple words, e.g. 'dark brown') within a list of tokens.
    :param target: a String of the feature/country. E.g. 'dark brown', 'Italian wine', etc.
    :param tokens: the List of tokens of the sentence to be looked at
    :return: all indices where a match was found
    """

    featur_tokens = target.split()

    indices = []
    i = 0
    while i < len(tokens):
        equal = True
        for j, ft in enumerate(featur_tokens):
            if i + j >= len(tokens) or tokens[i + j] != ft:
                equal = False
                break
        if equal:
            indices.append(i)
            i += len(featur_tokens)
        i += 1
    return indices



def find_hyper_hypo_pairs(s1, s2, tokens1, tokens2, is_print=False, search_only_nouns=True, filter_repeat_word=False):
    # find hypernym-hyponym pairs, so one is in s1 and the other in s2 and vise versa
    # Input:
    #   s1 and s2, two sentences to locate hyponym in one and hypernum at the other and vice versa
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (hypo, hyper_s_orig, "hypo->hyper", (hypo_ind, hyper_s_ind))
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form

    global singular_errors
    global excluded_classes

    both_in_p_and_h = 0

    pairs_p_h, pairs_h_p = {}, {}
    duplicates = ""
    not_printed = True
    words_hypernyms_s1, doc1, nouns1 = get_hypernym_candidates2(s1, is_print=False) # look for the last word in each NP and returns all of its hypernyms, for all senses
    words_hypernyms_s2, doc2, nouns2 = get_hypernym_candidates2(s2, is_print=False) #
    tokesn2_only_nouns = nouns2 if search_only_nouns else tokens2
    tokesn1_only_nouns = nouns1 if search_only_nouns else tokens1

    def update_d(d, values):
        hypo, hyper, direction, hypo_ind, hyper_ind = values
        if hyper not in d:
            d[hyper] = {'hypo':[hypo] * len(hypo_ind), 'direction':direction, 'hypo_ind':hypo_ind.copy(), 'hyper_ind':hyper_ind.copy()}
        elif hypo not in d[hyper]['hypo']:      # we only want each hypo once for each hyper
            d[hyper]['hypo'] += [hypo] * len(hypo_ind)
            d[hyper]['hypo_ind'] += hypo_ind
        assert len(d[hyper]['hypo'])==len(d[hyper]['hypo_ind']), f"d[hyper]['hypo']={d[hyper]['hypo']} but d[hyper]['hypo_ind']={d[hyper]['hypo_ind']}"
        return d


    # looking for hypo in s1 and hyper in s2
    for (hypo, hs) in words_hypernyms_s1:       # words_hypernyms_s1 = [('dog', ['Synset(animal.n.4)', 'Synset(mammal.n.1)'...]), ]
        hypo_ind = find_all_ind(hypo, to_singular(tokens1))
        # print('*** ' + hypo + ':')
        for hi in hs:
            hyper_orig = synset_name(hi)
            hyper_s = to_singular(hyper_orig)
            # # assert hyper_s == to_singular(hyper_s) , f"hypers='{hyper_s}' should be singular, but is seems to be plural"
            # ### TODO: bring back the assert above instead of the 'if' below
            # if hyper_s != to_singular(hyper_s) and hyper_s not in singular_errors:
            #     print(hyper_s)
            #     warnings.warn(f'hypernymy {hyper_s} (of {hypo}) seems to be plural')

            # print(hyper_s + ' -> ')
            # if s2.find(hyper_s) > -1 and s1.find9(hyper_s) < 0 and s2.find(hypo) < 0 and hyper_s not in excluded_classes:
            do_filter = filter_repeat_word and ((hyper_s in to_singular(tokesn1_only_nouns)) or (hypo in to_singular(tokesn2_only_nouns)))
            if hyper_s in to_singular(tokesn2_only_nouns) and not do_filter:   # looking for the hyper in s2
                hyper_s_ind = find_all_ind(hyper_s, to_singular(tokens2))
                if len(hyper_s_ind) > 0:
                    duplicates += "hyper appears twice in S2. "
                # hyper_s_orig = tokens2[hyper_s_ind[0]]
                # pairs_p_h.append((hypo, hyper_s, "hypo->hyper", hypo_ind, hyper_s_ind))
                pairs_p_h = update_d(pairs_p_h, (hypo, hyper_s, "hypo->hyper", hypo_ind, hyper_s_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (hypo, hyper_s))

    # looking for hypo in s2 and hyper in s1
    for (hypo, hs) in words_hypernyms_s2:
        hypo_ind = find_all_ind(hypo, to_singular(tokens2))
        # print('*** ' + hypo + ':')
        for hi in hs:
            hyper_orig = synset_name(hi)
            hyper_s = to_singular(hyper_orig)               # assert hyper_s == to_singular(hyper_s), f"hypers='{hyper_s}' should be singular, but is seems to be plural"
            ### TODO: bring back the assert above instead of the 'if' below
            # if hyper_s != to_singular(hyper_s) and hyper_s not in singular_errors:
            #     print(hyper_s)
            #     warnings.warn(f'hypernymy {hyper_s} (of {hypo}) seems to be plural')

            # print(hyper_s + ' -> ')
            # if s2.find(hyper_s) > -1 and s1.find9(hyper_s) < 0 and s2.find(hypo) < 0 and hyper_s not in excluded_classes:
            do_filter = filter_repeat_word and ((hyper_s in to_singular(tokesn2_only_nouns)) or (hypo in to_singular(tokesn1_only_nouns)))
            if hyper_s in to_singular(tokesn1_only_nouns) and not do_filter:
                hyper_s_ind = find_all_ind(hyper_s, to_singular(tokens1))
                if len(hyper_s_ind) > 1:
                    duplicates += "hyper appears twice in S1. "
                # hyper_s_orig = tokens1[hyper_s_ind[0]]
                # pairs_h_p.append((hypo, hyper_s, "hyper->hypo", hypo_ind, hyper_s_ind))
                pairs_h_p = update_d(pairs_h_p, (hypo, hyper_s, "hyper->hypo", hypo_ind, hyper_s_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (hypo, hyper_s))
    return (pairs_p_h, pairs_h_p), (tokens1, tokens2)



def find_hypernymy_pairs(s1, s2, tokens1, tokens2, is_print=False, search_only_nouns=True, filter_repeat_word=False, mode='noun'):
    # find hyponym-hypernym pairs in this order ONLY, so one is in s1 and the other in s2.
    # Input:
    #   s1 and s2, two sentences to locate hyponym in one and hypernyum at the other.
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (hypo, hyper_s_orig, "hypo->hyper", (hypo_ind, hyper_s_ind))
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form

    global singular_errors
    global excluded_classes

    both_in_p_and_h = 0

    s1 = re.sub(' +', ' ', s1)
    s2 = re.sub(' +', ' ', s2)
    pairs_p_h, pairs_h_p = {}, {}
    duplicates = ""
    not_printed = True
    words_hypernyms_s1, doc1, nouns1 = get_hypernym_candidates2(s1, is_print=False, mode=mode) # look for the last word in each NP and returns all of its hypernyms, for all senses
    words_hypernyms_s2, doc2, nouns2 = get_hypernym_candidates2(s2, is_print=False, mode=mode) #
    tokesn2_only_nouns = nouns2 if search_only_nouns else tokens2
    tokesn1_only_nouns = nouns1 if search_only_nouns else tokens1

    def update_d(d, values):
        hypo, hyper, direction, hypo_ind, hyper_ind = values
        if hyper not in d:
            d[hyper] = {'hypo':[hypo] * len(hypo_ind), 'direction':direction, 'hypo_ind':hypo_ind.copy(), 'hyper_ind':hyper_ind.copy()}
        elif hypo not in d[hyper]['hypo']:      # we only want each hypo once for each hyper
            d[hyper]['hypo'] += [hypo] * len(hypo_ind)
            d[hyper]['hypo_ind'] += hypo_ind
        assert len(d[hyper]['hypo'])==len(d[hyper]['hypo_ind']), f"d[hyper]['hypo']={d[hyper]['hypo']} but d[hyper]['hypo_ind']={d[hyper]['hypo_ind']}"
        return d


    # looking for hypo in s1 and hyper in s2
    for (hypo, hs) in words_hypernyms_s1:       # words_hypernyms_s1 = [('dog', ['Synset(animal.n.4)', 'Synset(mammal.n.1)'...]), ]
        hypo_ind = find_all_ind(hypo, to_singular(tokens1))
        # print('*** ' + hypo + ':')
        for hi in hs:
            hyper_orig = synset_name(hi)
            hyper_s = to_singular(hyper_orig)
            # # assert hyper_s == to_singular(hyper_s) , f"hypers='{hyper_s}' should be singular, but is seems to be plural"
            # ### TODO: bring back the assert above instead of the 'if' below
            # if hyper_s != to_singular(hyper_s) and hyper_s not in singular_errors:
            #     print(hyper_s)
            #     warnings.warn(f'hypernymy {hyper_s} (of {hypo}) seems to be plural')

            # print(hyper_s + ' -> ')
            # if s2.find(hyper_s) > -1 and s1.find9(hyper_s) < 0 and s2.find(hypo) < 0 and hyper_s not in excluded_classes:
            do_filter = filter_repeat_word and ((hyper_s in to_singular(tokesn1_only_nouns)) or (hypo in to_singular(tokesn2_only_nouns)))
            if hyper_s in to_singular(tokesn2_only_nouns) and not do_filter:   # looking for the hyper in s2
                hyper_s_ind = find_all_ind(hyper_s, to_singular(tokens2))
                if len(hyper_s_ind) > 0:
                    duplicates += "hyper appears twice in S2. "
                # hyper_s_orig = tokens2[hyper_s_ind[0]]
                # pairs_p_h.append((hypo, hyper_s, "hypo->hyper", hypo_ind, hyper_s_ind))
                pairs_p_h = update_d(pairs_p_h, (hypo, hyper_s, "hypo->hyper", hypo_ind, hyper_s_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (hypo, hyper_s))
    return pairs_p_h, (doc1, doc2)



def find_location_country_pairs(s1, s2, tokens1, tokens2, local_wiki, is_print=False,  filter_repeat_word=False, include_ORG=True):
    # find location-country pairs, so one is in s1 and the other in s2
    # Input:
    #   s1 and s2, two sentences to find a location in s1 and a country in s2
    #   filter_repeat_word - if the location or the country appear both in s1 and in s2, don't count that pair (since it's probably not a location-country phenomenon)
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (location, country, loc_ind, country_ind)
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form
    global singular_errors
    global excluded_classes

    pairs = {}
    duplicates = ""
    not_printed = True
    locations_s1 = get_location_candidates(s1, include_ORG) # returns all entities (using Spacy) that are LOC or GPE in lower()

    def update_d(d, values):
        location, country, location_ind, country_ind = values
        if country not in d:
            d[country] = {'location':[location] * len(location_ind), 'location_ind':location_ind.copy(), 'country_ind':country_ind.copy()}
        elif location not in d[country]['location']:      # we only want each location once for each country
            d[country]['location'] += [location] * len(location_ind)
            d[country]['location_ind'] += location_ind
        assert len(d[country]['location'])==len(d[country]['location_ind']), f"d[country]['location']={d[country]['location']} but d[country]['location_ind']={d[country]['location_ind']}"
        return d

    # looking for location in s1 and a matching country in s2
    for location in locations_s1:       # locations_s1 = ['kamakura', 'jerusalem']
        # countries = location2country(local_wiki, location)
        countries = location2country_or_state(local_wiki, location)
        location_ind = find_all_ind(location, tokens1)

        if location_ind == []: continue     # in case of location with multiple words, skip this location.
        for country in countries:
            do_filter = filter_repeat_word  and ((country in tokens1) or (location in tokens2))
            country_ind = find_all_country_ind(country, tokens2)
            if country_ind != [] and not do_filter:
                if len(country_ind) > 1:
                    duplicates += f"country {country} appears twice in S2. "
                pairs = update_d(pairs, (location, country, location_ind, country_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (location, country))
    save_local_wiki(local_wiki)
    return pairs


def find_trademark_country_pairs(s1, s2, tokens1, tokens2, local_wiki, is_print=False,  filter_repeat_word=False, include_ORG=True):
    # find company-country pairs, so one is in s1 and the other in s2
    # Input:
    #   s1 and s2, two sentences to find a company in s1 and a country in s2
    #   filter_repeat_word - if the company or the country appear both in s1 and in s2, don't count that pair (since it's probably not a company-country phenomenon)
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (company, country, loc_ind, country_ind)
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form
    global singular_errors
    global excluded_classes

    pairs = {}
    duplicates = ""
    not_printed = True
    companies_s1 = get_capitalized_candidates(s1)  # returns all potential entities in lower(), by taking all words with capitalized first letter

    def update_d(d, values):
        company, country, company_ind, country_ind = values
        if country not in d:
            d[country] = {'company':[company] * len(company_ind), 'company_ind':company_ind.copy(), 'country_ind':country_ind.copy()}
        elif company not in d[country]['company']:      # we only want each company once for each country
            d[country]['company'] += [company] * len(company_ind)
            d[country]['company_ind'] += company_ind
        assert len(d[country]['company'])==len(d[country]['company_ind']), f"d[country]['company']={d[country]['company']} but d[country]['company_ind']={d[country]['company_ind']}"
        return d

    def capi(word):
        return word[0].upper() + word[1:]

    def country2adj_func(country):
        if country in country2adj:
            return country2adj[country]
        else:
            return country

    # looking for company in s1 and a matching country in s2
    for company in companies_s1:       # companies_s1 = ['anobit', 'Israel']
        # countries = location2country(local_wiki, company)
        countries = location2country(local_wiki, company)
        company_ind = find_all_ind(company, tokens1)

        if company_ind == []: continue     # in case of company with multiple words, skip this company.
        for country in countries:
            do_filter = filter_repeat_word and ((country in tokens1) or (company in tokens2))
            country_ind = find_all_country_ind(country, tokens2)
            country_ind += find_all_country_ind(country2adj_func(capi(country)).lower(), tokens2)
            if country_ind != [] and not do_filter:
                if len(country_ind) > 1:
                    duplicates += f"country {country} appears twice in S2. "
                pairs = update_d(pairs, (company, country, company_ind, country_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (company, country))
    save_local_wiki(local_wiki)
    return pairs


def find_color_pairs(s1, s2, tokens1, tokens2, local_wiki_features, is_print=False,  filter_repeat_word=False):
    # find noun-country pairs, so one is in s1 and the other in s2
    # Input:
    #   s1 and s2, two sentences to find a noun in s1 and a country in s2
    #   filter_repeat_word - if the noun or the country appears both in s1 and in s2, don't count that pair (since it's probably not a noun-country phenomenon)
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (noun, country, loc_ind, country_ind)
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form
    global singular_errors
    global excluded_classes

    duplicates = ""
    not_printed = True
    get_hypernym_candidates2
    nouns, doc = pick_nouns_for_color(s1)   # Same principle as in hypernyms - picks the last noun in an NP constituency.
    pairs = {}
    # locations_s1  = get_location_candidates(s1, include_ORG) # returns all entities (using Spacy) that are LOC or GPE in lower()
    feature_types = ('color', 'shape', 'material')

    def update_d(d, values):
        noun, feature, noun_ind, feature_ind, feature_type = values
        if feature not in d:
            d[feature] = {'noun':[noun] * len(noun_ind), 'noun_ind':noun_ind.copy(), 'feature_ind':feature_ind.copy(), 'type': feature_type}
        elif noun not in d[feature]['noun']:      # we only want each noun once for each feature
            d[feature]['noun'] += [noun] * len(noun_ind)
            d[feature]['noun_ind'] += noun_ind
        assert len(d[feature]['noun'])==len(d[feature]['noun_ind']), f"d[feature]['noun']={d[feature]['noun']} but d[feature]['noun_ind']={d[feature]['noun_ind']}"
        return d

    ##TODO: I think that there is a bug with location of the period at end of sentence ==> 'LastWordToken.' instead of 'LastWordToken'
    # looking for noun in s1 and a matching feature in s2
    for noun in nouns:       # nouns = ['basketball', 'table']
        features_d = noun2features(local_wiki_features, noun)
        noun_ind = find_all_ind(noun, to_singular(tokens1))

        if noun_ind == []: continue     # in case of noun with multiple words, skip this noun.
        for feature_type in features_d:      # features_d = ((colors...), (shapes...), (material used...))
            if feature_type == 'color' and len(features_d['color']) > 1: continue      # ignore objects with more than one color
            for feature in features_d[feature_type]:
                feature = features_synonyms(feature, feature_type)      # converts to the synonym as appears in wikidata, if exists. e.g round -> sphere
                do_filter = filter_repeat_word and ((feature in tokens1) or (noun in to_singular(tokens2)))
                feature_ind = find_all_ind2(feature, tokens2)       # looks for the actual words of the feature in the second sentence (e.g. lookse for "dark brown" in s2)
                if feature_ind != [] and not do_filter:
                    if len(feature_ind) > 1:
                        duplicates += f"feature {feature} appears twice in S2. "
                    pairs = update_d(pairs, (noun, feature, noun_ind, feature_ind, feature_type))
                    if is_print:
                        if not_printed:
                            print('\n' + s1)
                            print(s2)
                            not_printed = False
                        else:
                            print("-----------Multiple----------")
                        print('*** Found in H:%s -> %s' % (noun, feature))
    save_local_wiki_features(local_wiki_features)
    return pairs


def word_peice_connected(tokens, input_ids):
    # Input: tokens of wordpiece tokenizer output and their id's
    # Output: the connected token list + new2old_ind converter
    # For example:
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

def search_class(class0, NLIFile, outputFile, start_ind=0, length=100):
    print("loading multinli ...")
    results = []
    sents = load_multinli(NLIFile, start_ind, length)
    founded =0
    with open(outputFile + 'dobj.txt', 'w', encoding="utf-8") as fw:
        for (s, _) in tqdm(sents, desc="searching"):
            nouns, doc = pick_nouns_for_hypernyms(s)
            for w in nouns:
                h = all_hypernyms(w)
                if h:
                    for hi in h:
                        if hi.name().find(class0)>-1:
                            print('\n' + w + ': ' + s)
                            results.append((w, s))
                            founded +=1
    return results, founded

def search_hyper_pairs(NLIFile, outputFile, start_ind=0, length=100):
    global excluded_classes

    print("loading multinli ...")
    sents = load_multinli(NLIFile, start_ind, length)
    founded =0
    with open(outputFile + 'dobj.txt', 'w', encoding="utf-8") as fw:
        for (p_in, h_in) in tqdm(sents, desc="searching"):
            p = word_tokenize(p_in)
            h = word_tokenize(h_in)
            for w in p:
                hypers = all_hypernyms(w)
                if hypers:
                    for hyper in hypers:
                        hyper = hyper.name().split('.')[0]
                        if hyper not in excluded_classes:
                            for w2 in h:
                                w2_l = lemmatizer.lemmatize(w2)
                                if hyper == w2_l:
                                    print('\nPremise: %s'%p_in)
                                    print('Hypothesis: %s'%h_in)
                                    print('%s -> %s'%(w, w2))
                                    founded +=1
    return founded



def search_mnli_hypernyms(NLIFile, outputFile, start_ind=0, length=100):
    global excluded_classes
    excluded_genres = ['telephone']
    print("loading multinli ...")
    results = []
    sents = load_multinli(NLIFile, start_ind, length * 2)   # loading twice as much as needed, as later we'll filter the results to about 70%
    # with open(outputFile + 'dobj.txt', 'w', encoding="utf-8") as fw:
    with open(outputFile + 'dobj.txt', 'w') as fw:
        for i, (s, _, genre) in enumerate(sents):
            if genre in excluded_genres: continue
            to_print = []
            nouns, doc = pick_nouns_for_hypernyms(s)
            hyponyms = []
            for w in nouns:
                h = all_hypernyms_names(w)
                line_s = ''
                hypernyms = []
                if h:
                    for hi in h:
                        # hi = hi.name().split('.')[0]
                        if hi not in excluded_classes:
                            hypernyms.append(hi)
                            line_s += ' ; ' + hi
                            # results.append((w, s))
                            # found +=1
                if hypernyms:
                    hyponyms.append((w, hypernyms, is_plural(w)))
                if line_s:
                    to_print.append(w + '=>' + line_s)
            if hyponyms:
                sent_ind = i + start_ind    # index of the sentence within the dataset
                results.append((s, genre, hyponyms, sent_ind))
                if len(results) >= length: break
            if to_print:
                print('%d. (%s)' % (i, genre) + s)
                for line in to_print:
                    print(line)
                print('\n-------------------------------------')
    return results

def hypernyms_results_to_csv(results, filename):
    rows = []
    for r in results:
        d = {}
        d['index'] = r[3]
        d['premise'] = r[0]
        d['genre'] = r[1]
        nb_hypos = min(len(r[2]), 5)
        for hypo_i in range(nb_hypos):
            d['hyponym'+str(hypo_i+1)] = r[2][hypo_i][0]
            d['hypernyms'+str(hypo_i+1)] = ' &nbsp|&nbsp '.join(r[2][hypo_i][1])
            d['plurality'+str(hypo_i+1)] = r[2][hypo_i][2]
        for hypo_i in range(nb_hypos, 5):
            d['hyponym'+str(hypo_i+1)] = ""
            d['hypernyms'+str(hypo_i+1)] = ""
            d['plurality'+str(hypo_i+1)] = ""

        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print('Wrote %d lines to %s'%(len(df),filename))

    cols = ['premise', 'hyponym1', 'hyponym2', 'hyponym3', 'hyponym4', 'hyponym5', 'hypernyms1', 'hypernyms2', 'hypernyms3', 'hypernyms4', 'hypernyms5', 'plurality1', 'plurality2', 'plurality3', 'plurality4', 'plurality']
    df = pd.DataFrame()

def all_synonyms_antonyms(word):
    synonyms = []
    antonyms = []

    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return set(synonyms), set(antonyms)

def all_senses(word):
    senses = []
    for syn in wn.synsets(word):
        senses.append((syn.lemmas()[0], syn.hypernym_paths()))
    return senses

def navigate_hypernyms(word, i=0):
    while i >= 0:
        print('Word = %s' % word)
        syns = wn.synsets(word)
        hypers = all_hypernyms(word)
        hypos = all_hyponyms(word)
        syns_s = 'Syns: '+ ' '.join(['(%d)%s' % (i, w.name()) for i, w in enumerate(syns)])
        hyper_s = 'Hypers:' + ' '.join(['(%d)%s' % (i, w.name()) for i, w in enumerate(hypers)])
        hypo_s = 'Hypos:' + ' '.join(['(%d)%s' % (i, w.name()) for i, w in enumerate(hypos)])
        print(syns_s)
        print(hyper_s)
        print(hypo_s)
        x = input().strip()
        if x[0]=='-':
            word = x[1:]
            i =0
        elif x.find('b') == 0:
            i = int(x[1:])
            word = hypos[i].name().split('.')[0]
        elif x.find('a') == 0:
            i = int(x[1:])
            word = hypers[i].name().split('.')[0]
        elif x.find('c') == 0:
            i = int(x[1:])
            word = syns[i].name().split('.')[0]
        else:
            i = int(x)
            word = hypers[i].name().split('.')[0]

def similar_words(input_word, k=5):
    # Format the input vector for use in the distance function
    # In this case we will artificially create a word vector from a real word ("frog")
    # but any derived word vector could be used
    p = np.array([nlp.vocab[input_word].vector])

    # Format the vocabulary for use in the distance function
    ids = [x for x in nlp.vocab.vectors.keys()]
    vectors = [nlp.vocab.vectors[x] for x in ids]
    vectors = np.array(vectors)

    # *** Find the closest word below ***
    closest_index = distance.cdist(p, vectors)
    sorted_ind = closest_index.argsort()[-3:][::-1]

    i = 0
    results = []
    while len(results) <= k:
        word_id = ids[sorted_ind[0,i]]
        output_word = nlp.vocab[word_id].text.lower()
        if output_word not in results:
            results.append(output_word)
        i += 1
    return results

def wordfreq(word):
    a = []
    syns = wn.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if l.name() == word:
                a.append(l.count())
    if a:
        return max(a)



def is_hyponym_by_hyper_sense(hypo, hyper_sense):
    # checks if the input 'word' is a hyponym of hyper_sense.
    hypo_l = lemmatizer.lemmatize(hypo.strip())
    for hypo_sense in wn.synsets(hypo_l):
        paths = hypo_sense.hypernym_paths()
        for path in paths:
            for cand_sense in path:
                if cand_sense == hyper_sense:
                    return True
                    # print(path)
    return None

def find_relevant_hyper_sense(hypo, hyper):
    # find the relevant hypernym sense for a given hypo
    for hyper_sense in wn.synsets(hyper):
        if is_hyponym_by_hyper_sense(hypo, hyper_sense):
            return hyper_sense
    return None


def similar_words_not_cohyponym(input_word, hypo, hyper, k=50, keep_plurality=True):
    # looking for most similar words to 'input_word' that are not co-hyponym of 'hypo' with respect to the common hypernym 'hyper'

    p = np.array([nlp.vocab[input_word].vector]) # Format the input vector for use in the distance function. In this case we will artificially create a word vector from a real word ("frog") but any derived word vector could be used
    # Format the vocabulary for use in the distance function
    ids = [x for x in nlp.vocab.vectors.keys()]
    vectors = [nlp.vocab.vectors[x] for x in ids]
    vectors = np.array(vectors)
    # *** Find the closest word below ***
    closest_index = distance.cdist(p, vectors)
    sorted_ind = closest_index.argsort()[-3:][::-1]

    i = -1
    results = []
    freq = []
    hypo, hyper = to_singular([hypo, hyper])
    hyper_sense = find_relevant_hyper_sense(hypo, hyper)   # hyper_sense is the relevant sense of 'hyper' for hyponym 'hypo'
    if hyper_sense is None:
        print("'%s' is not a hypernym for '%s' (find_relevant_hyper_sense(...) returned None)"%(hyper, hypo))
        return []
    output_word = ""
    while len(results) <= k:
        i += 1
        word_id = ids[sorted_ind[0,i]]
        prev_output_word = output_word
        output_word = nlp.vocab[word_id].text.lower()
        if output_word == prev_output_word: continue
        multiple_words = output_word.find('_') >= 0
        w_freq = wordfreq(output_word)
        is_noun = output_word not in pos_dict or 'NN' in pos_dict[output_word]
        is_hyponym = is_hyponym_by_hyper_sense(output_word, hyper_sense)
        if output_word not in results and not is_hyponym and not multiple_words and w_freq is not None and is_noun:
            results.append(output_word)
            freq.append(wordfreq(output_word))

    # freqind = np.array(freq).argsort()[::-1]
    # s_results = [results[i] for i in freqind]
    s_results = results
    return change_plurality(input_word, s_results) if keep_plurality else s_results

def is_plural(word):
    force_singular = ['process', 'bedclothes', 'address', 'compass', 'class', 'business', 'religious', 'judiciousness', 'attentiveness', 'access', 'cross',
                      'abbess', 'relatedness', 'illness', 'sameness', 'resoluteness', 'gas', 'kindness', 'orderliness', 'trustworthiness', 'activeness', 'inattentiveness',
                      'carelessness', 'loss']
    if not word.isalpha(): return word
    if word in force_singular:
        return False
    return not p.singular_noun(word) == False

def singular_noun_fixed(word):
    fixes = {'proces':'process', 'bedclothe':'bedclothes', 'due_proces': 'due_process', 'addres':'address', 'compas':'compass', 'clas':'class', 'busines':'business', '':''
        , 'religiou': 'religious', 'judiciousnes':'judiciousness', 'attentivenes':'attentiveness', 'acces':'access', 'cros':'cross'}
    if not word.isalpha(): return word
    sing = p.singular_noun(word)
    if sing in fixes:
        return fixes[sing]
    return sing

def to_singular(word_list):
    if type(word_list)==list:
        return [singular_noun_fixed(w) if (is_plural(w) and w.isalpha()) else w for w in word_list]
    else:
        return singular_noun_fixed(word_list) if ( word_list.isalpha() and is_plural(word_list)) else word_list

def to_plural(word_list):
    return [p.plural(w) if not is_plural(w) else w for w in word_list]

def change_plurality(reference_word, word_list):
    # changes the plurality of all words in word_list according to the reference_word plurality
    if is_plural(reference_word):
        return to_plural(word_list)
    else:
        return to_singular(word_list)

def sparq_wrapper(query_s, location=''):
    TimeOut = 300
    keep_trying = True
    tic = utils.Tic()
    i = 0
    sleeptime = 10

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query_s)
    sparql.setReturnFormat(JSON)

    while keep_trying:
        try:
            output = sparql.query().convert()
            keep_trying = False
            sleeptime = 10
        except Exception as e:
            i += 1
            time_elapsed = tic.toc(False)
            # print(f'failed. Time = {time_elapsed}, reason: {e}')
            if i%100 == 0:
                print('Time elapsed = ',time_elapsed)
            print('.', end='')
            if str(e).find('endpoint returned code 500 and response') >= 0:
                return []
            if str(e).find('Too Many Requests') >= 0:
                print('T, S=',sleeptime, '  Location = ', location)
                time.sleep(sleeptime)
                sleeptime += 10
                sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
                sparql.setQuery(query_s)
                sparql.setReturnFormat(JSON)
            keep_trying = time_elapsed < TimeOut
            if not keep_trying:
                print("Query TimeOut for location=", location)
                output = None
    return output

def get_wikidata_country_wrap(location):
    keep_trying = True
    tic = utils.Tic()
    time = 0

    while keep_trying:
        try:
            countries = wiki.get_wikidata_country(location)
            keep_trying = False
        except:
            time = tic.toc(False)
            print(f'failed. Time = {time}')
            keep_trying = time<15
    return countries

def country_synonyms(country):
    # converting country name to its synonyms as appear in Wikidata.
    #           E.g. 'US' -> 'the united states of america', united states -> 'the united states of america', ...
    multi2one = {'united states of america': ['united states of america', 'the united states of america', 'the united states', 'united states', 'us', 'usa', 'america', 'the us', 'the usa', 'u.s.', 'u.s', 'the u.s.', 'the u.s.a.', 'the u.s.a', 'u.s.a', 'u.s.a.'], 'united kingdom':['england','the united kingdom', 'united kingdom', 'uk', 'u.k.', 'the u.k.', 'great britain', 'britain']}
    # multi2one = {'united states of america': ['united states of america', 'the united states of america', 'the united states', 'united states', 'us', 'usa', 'america', 'the us', 'the usa', 'u . s .', 'u . s', 'the u . s.', 'the u . s . a .',
    #                                           'the u . s . a', 'u . s . a', 'u . s . a .'], 'united kingdom':['the united kingdom', 'united kingdom', 'uk', 'u . k .', 'the u . k .', 'great britain', 'britain']}

    for c in multi2one:
        if country in multi2one[c]:
            return c
    return country

def features_synonyms(feature, feature_type):
    # converting feature name to its synonyms as appear in Wikidata.
    #           E.g. 'sphere' -> 'round', ...
    d = {'color': {}, 'shape': {'round':'sphere'}, 'material': {}, }
    if feature in d[feature_type]:
        return d[feature_type][feature]
    return feature

def location2country(local_wiki, location):
    if location.lower() not in local_wiki:
        countries = get_wikidata_country_wrap(location)
        local_wiki[location] = [c.lower() for c in countries]
    return local_wiki[location]

def location2country_or_state(local_wiki, location):
    if location.lower() not in local_wiki:
        countries = wiki.get_wikidata_us_state_and_country(location)
        local_wiki[location] = [c.lower() for c in countries]    # it's a list for backward compatability
    return local_wiki[location]

def noun2features(local_wiki_features, noun):
    noun = to_singular(noun.lower())
    # if to_singular(noun) not in local_wiki_features:
    if noun not in local_wiki_features:
        features = wiki.get_wikidata_features(noun)
        local_wiki_features[noun] = features    # it's a list for backward compatability
    assert noun in local_wiki_features, f"the noun '{noun}' doesn't exist in the dictionary local_wiki_features with len {len(local_wiki_features)}"
    return local_wiki_features[noun]

def load_local_wiki():
    filename = 'local_location_wiki.p'
    return utils.load_pickle(filename)

def save_local_wiki(local_wiki):
    filename = 'local_location_wiki.p'
    utils.save_pickle(filename, local_wiki)

def load_local_wiki_features():
    filename = 'local_features_wiki.json'
    return utils.load_json(filename)

def save_local_wiki_features(local_wiki_features):
    filename = 'local_features_wiki.json'
    utils.save_json(filename, local_wiki_features)


def extract_locations_wikinews(filename, max_results=100000):
    df = pd.read_csv(filename)
    lines = df.iloc[:,1].values.tolist()
    i = 0
    for s in lines:
        if i > max_results: break
        doc = nlp(s)
        for e in doc.ents:
            if e.label_ in ['LOC', 'GPE']:
                loc = e.text.lower()
                print('')
                if loc in country_list:
                    print("%s (%s) is a country"%(e.text, e.label_))
                else:
                    print("%s (%s) is a location in %s " % (e.text, e.label_, wiki.get_wikidata_country(loc)))
                print(s)
                i += 1

def extract_locations_mnli(NLIFile, start_ind=0, length=1000):
    lines = load_multinli_w_label(NLIFile, start_ind, length)
    i = 0
    for (p, h, _, l) in lines:
        doc = nlp(p)
        for e in doc.ents:
            if e.label_ in ['LOC', 'GPE']:
                loc = e.text.lower()
                print('')
                if loc in country_list:
                    print("%s (%s) is a country"%(e.text, e.label_))
                else:
                    print("%s (%s) is a location in %s " % (e.text, e.label_, wiki.get_wikidata_country(loc)))
                print(p)

def extract_trademarks_from_mnli(NLIFile, start_ind=0, length=1000):
    lines = load_multinli_w_label(NLIFile, start_ind, length)
    sents = []
    for (p, h, genre, l) in lines:
        if genre == 'telephone':
            continue
        doc = nlp(p)
        trademarks = []
        tokens = [d.text for d in doc]
        for e in doc.ents:
            if e.label_ in ['ORG']:
                org = e.text
                country = wiki.get_wikidata_country(org)
                if org not in tokens:       #checking if org is one of the tokens. also filtering out multiple-word organizations.
                    continue
                ind = tokens.index(org)
                trademarks.append(org)
                s = ' '.join(tokens[:ind]) + ' **' + tokens[ind] + '** ' + ' '.join(tokens[ind+1:])
                d = dict(s=s, genre=genre, item=org, country=country)
                sents.append(d)
                print(f"({genre}) [{country}]) {s}")
    utils.dump_dict_to_csv('trademarks_premises.csv', sents)
    # print(s)
    # utils.save_json(json_filename, sents)
    # print("items(%d): %s"%(len(set(items)), str(set(items))))

    # print('')
    # print("%s (%s) is an ORG in %s " % (e.text, e.label_, country))
    # print(f"({genre}): {p}")
    # s = ' ' * (p.find(e.text) + len(genre) + 4) + '*' * len(e.text)
    # print(s)


def extract_items_for_features_from_mnli_old(NLIFile, start_ind=0, length=1000, feature_type='color'):
    artifact = wn.synsets('basketball')[1].hypernym_paths()[-1][4]
    json_filename = 'extracted_mnli_premises_for_' + feature_type + '.json'
    color_candidate_list = ['cider', 'Ariane', 'basketball', 'snowball', 'lingonberry', 'blackberry', 'raspberry', 'blueberry', 'feta', 'Bushwacker', 'milk', 'cherry', 'carrot', 'Clon', 'guabiroba', 'apricot', 'pineapple', 'cherry', 'apricot', 'Timorasso', 'baize', 'cola', 'boshiya', 'brandy', 'peridot', 'alabaster', 'elephant', '', 'Sirius', 'Greenstar', 'Eztika', 'Orgasm', 'Aviation', 'Appletini', 'cosmopolitan', 'tizana', 'Brunlok', 'Styrodur', 'Styrofoam', 'steel', 'Lagrein', 'Pelaverga', 'Midori', 'armstrongite', 'Chaux', 'taramite', 'clavus', 'cretonne', 'wakakusa', 'mohair', 'bombazine']
    color_candidate_list2 = ['drink', 'ball', 'strawbert', 'tomato', 'banana', 'cucumber', 'cheese', 'grape', 'melon', 'mango', 'cake', 'cola', 'hat', 'shirt']
    shape_condidate_list = ['bagel', 'Baranki', 'baseball', 'bolster', 'box', 'Cambozola', 'card', 'carrot', 'coil', 'dice', 'doughnut', 'fiber', 'flag', 'globe', 'marble', 'mug', 'napkin', 'porthole', 'rod', 'spring', 'string', 'wheel', '', 'anelloni', 'asparaginase', 'atoll', 'Bethmale', 'bevel', 'chiyogami', 'corniform', 'counterbore', 'crescent', 'croissant', 'cromlech', 'dolmen', 'Filipinos', 'Fougerus', 'hose', 'ikele', 'jingle', 'kanga', 'kazan', 'Ledikeni', 'nanotube', 'palet', 'Pannenkoeken', 'rack', 'Rajbhog', 'Saingorlon', 'salers', 'Slinky', 'stroopwafel']
    color_candidate_list_lower = [c.lower() for c in color_candidate_list]
    shape_condidate_list_lower = [c.lower() for c in shape_condidate_list]
    item_list = {'color': color_candidate_list_lower, 'shape': shape_condidate_list_lower, 'color2':color_candidate_list2}
    lines = load_multinli_w_label(NLIFile, start_ind, length)
    items, sents = [], {}
    c = 0
    for i, (p,h,_,l) in enumerate(lines):
        if i%100==0: print('-- ',i,' --')
        nouns, doc = pick_nouns_for_hypernyms(p)
        tokens = [d.text for d in doc]
        for t in nouns:
            # item_single = to_singular(t.lower())
            item_single = t
            # if item_single in item_list[feature_type]:

            if is_hyponym_by_hyper_sense(item_single, artifact):
                c += 1
                ind = find_all_ind(t, to_singular(tokens))[0]
                items.append(item_single)
                if item_single in sents:
                    sents[item_single].append(' '.join(tokens[:ind]) + ' **' + tokens[ind] + '** ' + ' '.join(tokens[ind+1:]))
                else:
                    sents[item_single] = [' '.join(tokens[:ind]) + ' **' + tokens[ind] + '** ' + ' '.join(tokens[ind + 1:])]
                s = ' '.join(tokens[:ind]) + ' **' + colored(tokens[ind], 'red') + '** ' + ' '.join(tokens[ind+1:])
                print(s)
                utils.save_json(json_filename, sents)
                print("items(%d): %s"%(len(set(items)), str(set(items))))



def extract_items_for_color_from_mnli(NLIFile, start_ind=0, length=1000, hypernym_groups=None):

    def find_matching_hypernym(item_single, hypernym_groups):
        ## checks if the item is an hyponym of any of the hypernym-groups

        # the name of the synsets below much match the keys in hypernym_groups
        artifact = wn.synsets('basketball')[1].hypernym_paths()[-1][4]
        animal = wn.synsets('elephant')[0].hypernym_paths()[1][6]
        cloth = wn.synsets('pants')[0].hypernym_paths()[0][7]
        drink = wn.synsets('beer')[0].hypernym_paths()[0][5]
        food = wn.synsets('watermelon')[1].hypernym_paths()[1][4]
        fabric = wn.synsets('baize')[0].hypernym_paths()[0][-2]
        material = wn.synsets('peridot')[0].hypernym_paths()[0][4]
        object = wn.synsets('ball')[0].hypernym_paths()[0][2]
        vehicle = wn.synsets('car')[0].hypernym_paths()[1][-5]

        for group in hypernym_groups:
            if is_hyponym_by_hyper_sense(item_single, eval(group)):
                return group
        return None

    MaxTokens = 30
    if hypernym_groups is None:
        hypernym_groups = {'animal': 3, 'cloth': 6, 'drink': 42, 'food': 57, 'fabric': 9, 'material': 21, 'object': 6}
    hypernym_groups_c = dict(animal=0, cloth=0, drink=0, food=0, fabric=0, material=0, object=0, vehicle=0)
    lines = load_multinli_w_label(NLIFile, start_ind, length=22000)
    sents = []
    c = 0
    excluded_genres = ['telephone']

    for i, (p ,h, genre, l) in enumerate(lines):
        if genre in excluded_genres:
            continue
        if i % 100 == 0: print('-- ', i, ' --')
        if c > length:
            break
        nouns, doc = pick_nouns_for_hypernyms(p)
        tokens = [d.text for d in doc]
        if len(tokens) > MaxTokens:
            continue
        for item_single in nouns:
            matching_hypernym = find_matching_hypernym(item_single, hypernym_groups)
            if matching_hypernym:
                if hypernym_groups_c[matching_hypernym] > hypernym_groups[matching_hypernym] * 10:
                    continue
                c += 1
                hypernym_groups_c[matching_hypernym] += 1
                ind = find_all_ind(item_single, to_singular(tokens))[0]
                s = ' '.join(tokens[:ind]) + ' **' + tokens[ind] + '** ' + ' '.join(tokens[ind + 1:])
                s_print = '[' + matching_hypernym + '] ' + ' '.join(tokens[:ind]) + ' **' + colored(tokens[ind], 'red') + '** ' + ' '.join(tokens[ind+1:])
                print(s_print)
                d = dict(s=s, genre=genre, noun=item_single, group=matching_hypernym)
                sents.append(d)
                # print(f"({genre}) [{item_single}]) {s}")
                break       # one item per sentence in enough.
    utils.dump_dict_to_csv('color_premises(new_format).csv', sents)



def extract_items_for_hypernymy_from_mnli(NLIFile, start_ind=0, length=20, hypernym_groups=None, max_premise_len=1000):

    def find_relevant_synsets(hypernym, example_hyponym):
        hypernym, example_hyponym = to_singular([hypernym, example_hyponym])
        sense = find_relevant_sense(wn.synsets(hypernym), example_hyponym)
        return sense

    def find_matching_hypernym(item_single, hypernym_groups):
        ## checks if the item is an hyponym of any of the hypernym-groups

        # the name of the synsets below much match the keys in hypernym_groups
        vegetable = find_relevant_synsets(hypernym='vegetable', example_hyponym='onion')
        fruit = find_relevant_synsets(hypernym='fruit', example_hyponym='banana')
        tree = find_relevant_synsets(hypernym='tree', example_hyponym='pine')
        flower = find_relevant_synsets(hypernym='flower', example_hyponym='daisies')
        mammal = find_relevant_synsets(hypernym='mammal', example_hyponym='dog')
        insect = find_relevant_synsets(hypernym='insect', example_hyponym='ant')
        bird = find_relevant_synsets(hypernym='bird', example_hyponym='parrot')
        fish = find_relevant_synsets(hypernym='fish', example_hyponym='shark')
        medicine = find_relevant_synsets(hypernym='medicine', example_hyponym='styptic')
        hormone = find_relevant_synsets(hypernym='hormone',  example_hyponym='epinephrines')
        wine = find_relevant_synsets(hypernym='wine', example_hyponym='Amontillado')
        liquor = find_relevant_synsets(hypernym='liquor', example_hyponym='whisky')
        fabric = find_relevant_synsets(hypernym='fabric', example_hyponym='towel')
        material = find_relevant_synsets(hypernym='material', example_hyponym='peridot')

        # structure = find_relevant_synsets(hypernym='structure', example_hyponym='resort')
        # adornment = find_relevant_synsets(hypernym='adornment', example_hyponym='jewelry')

        for group in hypernym_groups:
            if is_hyponym_by_hyper_sense(item_single, eval(group)):
                return group
        return None

    MaxTokens = 30
    MaxGroupSize = 100
    excluded_words = ['man', 'product', 'king', 'heart', 'ground', 'coffee', 'world', 'date', 'spot', 'bay', 'complex', 'worker','key', 'emperor',
                      'spike', 'empire','keys', 'poll', 'application', 'stock', 'application', 'ear', 'skin', 'end', 'soldier', 'hair', 'port', 'application', 'empire', 'powder']
    # if hypernym_groups is None:
    #     hypernym_groups = {'vegetable': 10, 'fruit': 6, 'tree': 42, 'flower': 57, 'mammal': 9, 'insect': 21, 'bird': 6}
    # hypernym_groups_c = dict(vegetable=0, fruit=0, tree=0, flower=0, mammal=0, insect=0, bird=0, fish=0,
    #                          medicine=0, hormone=0, wine=0, liquor=0, fabric=0, material=0)
    hypernym_groups_c = dict(medicine=0, hormone=0)
    lines = load_multinli_w_label(NLIFile, start_ind, length=22000)
    sents = []
    c = 0
    excluded_genres = ['telephone']

    for i, (p ,h, genre, l) in enumerate(lines):
        if genre in excluded_genres or len(p) > max_premise_len:
            continue
        if i % 100 == 0: print('-- ', i, ' --')
        if c % 100 == 0: print('++ c=', c, ' ++')
        if c % 10 == 0:
            utils.dump_dict_to_csv('hypernym_premises_new.csv', sents)
        if c > length:
            break
        nouns, doc = pick_nouns_for_hypernyms(p)
        tokens = [d.text for d in doc]
        if len(tokens) > MaxTokens:
            continue
        for item_single in nouns:
            if item_single in excluded_words:   # ignoring 'man' as a mammal, etc.
                continue
            matching_hypernym = find_matching_hypernym(item_single, hypernym_groups_c)
            if matching_hypernym:
                if hypernym_groups_c[matching_hypernym] > MaxGroupSize:
                    continue
                c += 1
                hypernym_groups_c[matching_hypernym] += 1
                hypo = to_plural([item_single])[0]
                hypo_singular = item_single
                loc = p.find(hypo)
                if loc < 0:
                    hypo = hypo_singular
                    loc = p.find(hypo)
                s = p[:loc] + ' **' + hypo + '**' + p[loc + len(hypo):]
                s_print = '(%s) '%matching_hypernym + p[:loc] + ' **' + colored(hypo, 'red') + '**' + p[loc + len(hypo):]
                # s_print = '[' + matching_hypernym + '] ' + ' '.join(tokens[:ind]) + ' **' + colored(tokens[ind], 'red') + '** ' + ' '.join(tokens[ind+1:])
                print(s_print)
                d = dict(s=s, genre=genre, noun=item_single, group=matching_hypernym)
                sents.append(d)
                # print(f"({genre}) [{item_single}]) {s}")
                break       # one item per sentence in enough.
    utils.dump_dict_to_csv('hypernym_premises_new.csv', sents)




def get_location_candidates(sentence, include_ORG=True):
    locations = []
    labels_to_include = ['LOC', 'GPE', 'ORG'] if include_ORG else ['LOC', 'GPE']
    doc = nlp(sentence)
    for e in doc.ents:
        if e.label_ in labels_to_include and e.text.lower() not in country_list and e.text.lower().find('.')==-1 and e.text.isalpha():      # not '.' in the location & if can't be a country name
            locations.append(e.text.lower())
    return locations

def get_capitalized_candidates(sentence):
    locations = []
    doc = nlp(sentence)
    for d in doc:
        if d.pos_ in ['PROPN', 'NOUN', 'X', 'ADJ'] and any([c.isupper() for c in d.text]):
            locations.append(d.text.lower())
    return locations

def search_duplicated_hypernymy_classes_json():
    filename = '/home/nlp/ohadr/PycharmProjects/ExtEmb_src/data/amturk_data/Hypernymy_New_datasets/json_converted_from_csv/all_hypernymy_examples_229p.json'
    examples = utils.load_json(filename)

    pairs = dict(insect='fish', fish='mammal', mammal='bird', bird='insect', wine='liquor', liquor='win', medicine='hormone', hormone='medicine', tree='flower', flower='tree', material='fabric', fabric='material')

    items = {}
    no_hyper = []
    double_hyper = []
    for hyper in pairs:
        item_set = set([e['metadata']['item'] for e in examples if e['metadata']['hypernym'] == hyper])
        items[hyper] = item_set

    for hyper in pairs:
        non_hyper = pairs[hyper]
        for m in items[hyper]:
            hypernym_list = list_hypernyms2([m], is_print=False, noun_only=True)
            if not hypernym_list:
                print(f"'{m}' has no hypernyms")
                no_hyper.append((m, hyper))
            else:
                all_item_hypers = [a.name() for a in hypernym_list[0][1]]
                for h in all_item_hypers:
                    if non_hyper in h:
                        print(f'{non_hyper} {m} is a {hyper}')
                        double_hyper.append((m, hyper))
                        pass
    return no_hyper, double_hyper


country_list = utils.load_pickle("country_list.p")

if __name__ =='__main__':
    # NLIFile = 'multinli_1.0_dev_matched.jsonl'
    NLIFile = 'multinli_1.0_train.jsonl'
    OutputFile = 'hypernyms.txt'
    mnli_csv_filename = 'mnli_hypernyms.csv'
    wikinews_filename = 'wikinews/wikinews.dev.full.csv'

    all_hypernyms2('software')
    ### example for extracting hyponyms for hypernym classes
    # all_hyponyms_by_freq('vegetable', 'eggplant')

    ### looking for ambiguous hyponyms / duplicated hypernyms per item:
    # search_duplicated_hypernymy_classes_json()
    # exit(0)

    ### extracting paris from MNLI
    # extract_items_for_hypernymy_from_mnli(NLIFile, length=50, start_ind=130000, max_premise_len=110)
    # # extract_items_for_color_from_mnli(NLIFile, length=450, start_ind=43000, hypernym_groups={'object': 50})
    # # # extract_trademarks_from_mnli(NLIFile, start_ind=6000, length=16000)
    # exit(0)

    # for debug:
    # find_hyper_hypo_pairs('i see dogs  and an animals outside and other dogs inside', 'the animal dogs is playing outside in the sand ', )
    # find_hyper_hypo_pairs('The authorities believe that the bounty hunters used their profession as a cover for armed crime.', 'It is believed that the bounty hunters are using their cover business to cover up the fact that they\'re committing armed robbery.', )
    common = ['lentil', 'corn', 'sapling', 'orchid', 'wallflower', 'flea', 'pheasant', 'flatfish', 'salve', 'noradrenaline', 'Manzanilla', 'grappa', 'tarpaulin', 'antibody']
    print([wordfreq(c) for c in common])
    local_wiki = load_local_wiki()
    # local_wiki_features = load_local_wiki_features()
    # local_wiki = {}
    local_wiki_features = {}
    # get_wikidata_country_wrap('paris')

    ### For debug Hypo-Hyper pairs :
    # debug = 'trademark-country'
    debug = 'hypernymy'
    # debug = 'find_county_ind'
    if debug == 'hypo-hyper':
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # a = "i see other animal inside with a furniture"
        # b = "the frog and the cats playing outside in the sand on chair"
        # a = "the authorities believe that the bounty hunters used their profession as a cover for armed crime . "
        # b = "it is believed that the bounty hunters are using their cover business to cover up the fact that they're committing armed housebreaking ."
        a = "nor have the rich maintained that audible indicator , that quasi - english , quasi - lockjawed accent that the swells all had in depression - era works . "
        b = "swells often made examples during the depression , in which you could hear their quasi - english , quasi - lockjawed accents that have been more or less maintained by the rich . "
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = word_peice_connected(tokens_a, [1] * len(tokens_a))[0], word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs, docs = find_hyper_hypo_pairs(a, b, tokens_a_new, tokens_b_new, False)
        pairs, docs = find_hyper_hypo_pairs(a, b, tokens_a_new, tokens_b_new, False, filter_repeat_word=True)
    elif debug == 'hypernymy':
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # a = 'So the question of the   chlorothiazide was finally abandoned , and the Coroner proceeded with his task . '
        # b = 'The hormone had been discussed , and the coroner moved on to more business that was pertinent to his task . '
        a = "Let 's see now ... Helms , he had akvavit ; so did Stevens ."
        b = "Helms has never had any liquor in his life . "
        # a = 'How do you know if your daughter is on  adrenosterones?'
        # b = 'Females can be on hormones.'
        a = "If you 're not in the mood for bubbly with your meal , have no qualms about ordering something else instead ."
        b = "If you 'd rather drink something besides wine to go with your meal , you can order orange juice . "
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = word_peice_connected(tokens_a, [1] * len(tokens_a))[0], word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs, docs = find_hypernymy_pairs(a, b, tokens_a_new, tokens_b_new, False, filter_repeat_word=True, mode='include_propn')
    elif debug == 'location-country':
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # a = "if you're coming from Brittany, you should check your GPS coordinates."
        # b = "I recommended you to check your GPS coordinates because of what city in America you were coming from."
        ## can't located Porto with Spact NER:
            # a = "although Porto has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment."
            # b = "Portugal has many pubs, bars, and discos."
        # a = "Although Trussville has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment."
        # b = "Alabama has many pubs, bars, and discos."
        a = "Although Trussville has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment."
        b = "Alabama has many pubs, bars, and discos."
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = word_peice_connected(tokens_a, [1] * len(tokens_a))[0], word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_location_country_pairs(a, b, tokens_a_new, tokens_b_new, local_wiki, include_ORG=True)
    elif debug == 'features':
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        a = "and um they have little bar there so we sit there and um sipping on some appletinis before we get to eat"
        b = "I sipped on a bright green cocktail"
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = word_peice_connected(tokens_a, [1] * len(tokens_a))[0], word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_color_pairs(a, b, tokens_a_new, tokens_b_new, local_wiki_features, is_print=True)
    elif debug == 'trademark-country':
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        a = "Cotral leather goods remain unequalled ."
        b = "There is an Italian company that produces leather goods."
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = word_peice_connected(tokens_a, [1] * len(tokens_a))[0], word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_trademark_country_pairs(a, b, tokens_a_new, tokens_b_new, local_wiki)

    elif debug == 'find_county_ind':
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        s = 'while I love Israel, the u.s.a. is a great place to leave'
        t = tokenizer.tokenize(s)
        ind = find_all_country_ind('united states of america', t)

    exit(0)
    all_hypernyms2('example', True)
    # all_hypernyms_names('growth')
    all_hyponyms_by_freq('products', 'movies')
    # multinli2hypernums(NLIFile, OutputFile, start_ind=60, length=100)
    search_multinli_hyper_pairs_test(NLIFile, OutputFile, start_ind=1060, length=5000)
    # search_multinli_hyper_pairs(NLIFile, OutputFile, start_ind=1060, length=1000)
    # res, founded = search_class('object',NLIFile, OutputFile, start_ind=10000, length=1000)
    # similar_words_not_cohyponym('generals', 'generals', 'workers')
    # print(similar_words_not_cohyponym('religion', 'religion', 'belief'))

    res0 = similar_words_not_cohyponym('dog', 'dog', 'animal')
    print(res0)
    # print(all_hypernyms_names('failures'))
    ### Creating csv file with hypo-hyper candidates for AMT
    res = search_mnli_hypernyms(NLIFile, OutputFile, start_ind=1000, length=100)
    df = hypernyms_results_to_csv(res, mnli_csv_filename)

    # founded = search_hyper_pairs(NLIFile, OutputFile, start_ind=10000, length=1000)
    # all_hyponyms('professional', 'attorney')
    # all_hyponyms_by_freq('professionals', 'attorney')
    # similar_words_not_cohyponym('attorney', 'attorney', 'professionals')


    # print(founded)
    pass


    # navigate_hypernyms('dog')

    pass
    # print(wn.synsets("horse")[0][0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()).hypernyms()
    save_local_wiki(local_wiki)

