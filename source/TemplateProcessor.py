import re
import os
from utils import softmax
import utils as ut
import numpy as np
import random
import json


"""
Template types:

*i* *have* *a* *brown* *dog* with *fur*
*i* *have* *a* *dog* with *fur*
entailment
*i* *have* *fur*
neutral
*i*: <a1: John; the kid;>; <a2: They;>; <a3:  the boys; the girls;>
*have*: <a1: has; plays with;>;<a2, a3, a4 : have; play with; look at>
*a*: <b1,b3: a>; <b2: "">
*brown*: brown; small; 
*dog*: <b1: dog; horse; >; <b2: dogs; horses;>;     <b3: kid; girl>
*fur*: <b1, b2: fur; a small nose; stripes> ;       <b3: ""; a hat; a bag>;

This will generate 240 Entailment pairs, with the first sentence as a Premise and the Second as a Hypothesis, and another 240 Neutral with the 3rd sentence as Hypothesis (same Premise)
<a1, b1, b2: word1; word2;...> - denotes belonging of both word1 and word2 to groups a1, b1, b2. 
    Now, only words that belongs to an 'a' group must be of type 'a1', and words that belong to 'b' group must
    belong to type 'b1' or 'b2'. E.g. from the template above, there can only appear a combination of 'a dog' and 'a horse', 
    but never 'a dogs', since 'a' belongs to groups 'b1' and 'b3' but 'dogs' belongs to 'b2'. 
    In the same way, 'a dog' or 'dogs' will never appear with 'a hat' or 'a bag'.
    'browb' and 'small' have no restrictions. 
"""


def inst2str(instances):
    prev_groups = instances[0][1]
    if prev_groups:
        s = '<' + ','.join(prev_groups) + ': '
    else:
        s = ''
    for inst, groups in instances:
        inst_t = inst if inst else '\"\"'
        if groups == prev_groups:
            s += inst_t + '; '
        else:
            prev_groups = groups
            s = s[:-2]
            s += '>; <' + ','.join(prev_groups) + ': ' + inst_t + '; '
    if prev_groups:
        s = s[:-2] + '>'
    return s

def parse_set_title(s, template_filename = ''):
    """ looking for a string of the form # Set Name: S2; Lexical:S1-Same; Syntax:S1-different-adding RC ..."""
    set_d = {}
    template_filename = template_filename.split('/')[-1]    # removing the path
    EP_from_filename = re.search(r'.*?(?=[_])', template_filename).group() if template_filename else ''
    set_name_from_filename = re.search(r'(?<=_).*?(?=[\.])', template_filename).group() if template_filename else ''
    set_name_g = re.search(r'(?<=Set Name:).*?(?=[;*#])', s, re.IGNORECASE)
    if set_name_g:  # if 'set name' is in the title it means it describes the set. parse accordingly
        lexical = re.search(r'(?<=Lexical:).*?(?=[;*#])', s, re.IGNORECASE)
        syntax = re.search(r'(?<=Syntax:).*?(?=[;*#])', s, re.IGNORECASE)
        other = re.search(r'(?<=Other:).*?(?=[;*#])', s, re.IGNORECASE)
        # set_d['name'] = set_name_g.group().strip()
        set_d['name'] = set_name_from_filename.strip() if template_filename else set_name_g.group().strip()     # taking the name from the filemname if possible. otherwise- from the file
        set_d['lexical'] = lexical.group().strip() if lexical else ''
        set_d['syntax'] = syntax.group().strip() if syntax else ''
        set_d['EP'] = EP_from_filename
        set_d['other'] = other.group().strip() if other else ''
    return set_d

def parse_template_title(s):
    # if 'set name' isn't in the title it means it describes the template. parse accordingly
    # looks for 'Tense:' or 'Syntax:' or 'Template:'. If it can't find it, it returns {}
    if s.lower().find('set name:') > 0: return {}
    tmp_d = {}
    tense = re.search(r'(?<=Tense:).*?(?=[;*#])', s, re.IGNORECASE)
    syntax = re.search(r'(?<=Syntax:).*?(?=[;*#])', s, re.IGNORECASE)
    verb = re.search(r'(?<=Verb:).*?(?=[;*#])', s, re.IGNORECASE)
    other = re.search(r'(?<=Other:).*?(?=[;*#])', s, re.IGNORECASE)
    id = re.search(r'(?<=Template:).*?(?=[;*#])', s, re.IGNORECASE)
    if not (tense or syntax or id): return {}
    tmp_d['tense'] = tense.group().strip() if tense else ''
    tmp_d['syntax'] = syntax.group().strip() if syntax else ''
    tmp_d['verb'] = verb.group().strip() if verb else ''
    tmp_d['other'] = other.group().strip() if other else ''
    tmp_d['id'] = id.group().strip() if id else ''
    return tmp_d

def parse_args_title(s):
    # looks for 'args:'. agrs line shuold look like: "# args:  (*in contrast* = thus)  (*have delivered* = have passed)  (*them* = the big famous focus group)"
    if s.lower().find('args:') < 0 or s.lower().find('set name:') > 0 or s.lower().find('template:') > 0: return {}
    tmp_d = {}
    args_list = re.findall(r'(?<=<<).*?(?=[>>])', s, re.IGNORECASE)
    for arg_str in args_list:
        if not re.search(r'(?<=\*).*?(?=[\*])', arg_str, re.IGNORECASE):
            print("couldn't find arg for %s for arglist %s"%(arg_str, args_list))
        arg = re.search(r'(?<=\*).*?(?=[\*])', arg_str, re.IGNORECASE).group().strip()
        inst = re.search(r'(?<=\=).*', arg_str, re.IGNORECASE).group().strip()
        tmp_d[arg] = inst
    return tmp_d


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, info=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.info = info
        self.pred_before = ''
        self.probs_before = []
        self.conf_before = None
        self.good_before = None
        self.pred_after = ''
        self.probs_after = []
        self.good_after = None
        self.conf_after = None

    def update_pred(self, before_or_after, pred, probs):
        assert probs.shape == (3,), 'probs must be of shape (3,)'
        probs = list(probs)
        if before_or_after == 'before':
            self.pred_before = pred
            self.probs_before = probs
            self.conf_before = np.max(softmax(probs))*100
            self.good_before = int(pred==self.label)
        elif before_or_after == 'after':
            self.pred_after = pred
            self.probs_after = probs
            self.conf_after = np.max(softmax(probs)) * 100
            self.good_after = int(pred==self.label)
        else:
            raise Exception('before_or_after must be either before or after')

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def print_all_vars(self):
        print("text_a = %s"%self.text_a)
        print("text_b = %s"%self.text_b)
        print("label = %s"%self.label)
        print("info = ", self.info)
        print("pred_before = %s"%self.pred_before)
        print("probs_before = ",self.probs_before)
        if not self.conf_before==None:
            print("conf_before = %1.2f"%self.conf_before)
            print("good_before = %d"%self.good_before)
        print("pred_after = %s"%self.pred_after)
        print("probs_after = ",self.probs_after)
        if not self.conf_after == None:
            print("conf_after = %1.2f"%self.conf_after)
            print("good_after = %d"%self.good_after)


class TemplateProcessor(object):
    """Base class for data converters for sequence classification data sets."""


    def get_templates(self, templates_dir, EP, setname, dest_dir ='', dest = '', is_label_balancing = False):
        """

        :param templates_dir:
        :param EP: same as the EP in the set filename
        :param setname: same as the EP in the set filename
        :param dest: destination of the generated examples file

        :return: train_examples, train_templates (or test) - both lists
        train_examples are an InputExample object:
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.label = label
            self.info = info
        while train_exapmles.info looks like:
             {'set': {'name': 'S2',   'lexical': 'S1-Same',   'syntax': 'S1-different-adding RC',   'other': ''},
                 'template': {'tense': 'Future',    'syntax': 'V V V NP',    'id': '2'},
                 'hypothesis_id': '0'}
        """

        def calc_label_balancing(examples):
            d = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
            for e in examples:
                d[e.label] += 1
            print("Examples by labels: %s" % d)
            min_val = min(list(d.values()))
            label_balancing = {key: 1. * min_val / d[key] for key in d}
            return label_balancing

        def balance_labels(examples, label_balancing):
            examples_out = []
            for e in examples:
                if random.random() < label_balancing[e.label]:
                    examples_out.append(e)
            return examples_out



        if EP=='Numbers':
            # Balancing the labels from E:4/22 N:6/22 E:12/22 to 1/3 for all. Formula: total number of hypothesis (4+6+12=22) divided by (number of hypo per each label * 3). E: 22/(4*3) - 1.833
            label_balancing = {'entailment': 1.8333/1.8333, 'neutral': 1.2222/1.8333, 'contradiction': 0.6111/1.8333} if is_label_balancing else None

            if setname.find('R1-9') > 0:
                Numbers_EP_Range = (1,9)
            elif setname.find('R10-99') > 0:
                Numbers_EP_Range = (10, 99)
            elif setname.find('R100-199') > 0:
                Numbers_EP_Range = (100, 199)
            elif setname.find('R200-299') > 0:
                Numbers_EP_Range = (200, 299)
            elif setname.find('R400-499') > 0:
                Numbers_EP_Range = (400, 499)
            elif setname.find('R100-999') > 0:
                Numbers_EP_Range = (100, 999)
            elif setname.find('R30-50') > 0:
                Numbers_EP_Range = (30, 49)
            elif setname.find('R60-80') > 0:
                Numbers_EP_Range = (60, 79)
            elif setname.find('R1k-10k') > 0:
                Numbers_EP_Range = (1000, 9999)
            elif setname.find('R1-99') > 0:
                Numbers_EP_Range = (1, 99)
            elif setname.find('R1-499') > 0:
                Numbers_EP_Range = (1, 499)
            elif setname.find('R500-999') > 0:
                Numbers_EP_Range = (500, 999)
            else:
                assert setname.lower().find('r')<0, "Couldn't find the matching range for set %s"%(setname)
                Numbers_EP_Range = (1, 999)
        else:
            Numbers_EP_Range = []

        if EP=='Datives':
            # Balancing the labels from E:4/22 N:6/22 E:12/22 to 1/3 for all. Formula: total number of hypothesis (4+6+12=22) divided by (number of hypo per each label * 3). E: 22/(4*3) - 1.833
            label_balancing = {'entailment': 0.5, 'contradiction': 1, 'neutral': 1,} if is_label_balancing else None    #!!!! Note that I dilute the total number of exampels per set, since 0.5*2 + 1*1 = 2 instead of 3 hypothesys originally (ent, ent and cont)

        # if EP=='Hypernymy':
        #     # Balancing the labels from E:4/22 N:6/22 E:12/22 to 1/3 for all. Formula: total number of hypothesis (4+6+12=22) divided by (number of hypo per each label * 3). E: 22/(4*3) - 1.833
        #     label_balancing = {'entailment': 1, 'contradiction': 0.5, 'neutral': 1,} if is_label_balancing else None    #!!!! Note that I dilute the total number of exampels per set, since 0.5*2 + 1*1 = 2 instead of 3 hypothesys originally (ent, ent and cont)
        #
        #     if setname.find('_A') > 0:
        #         Hypernymy_Group = 'A'
        #     elif setname.find('_B') > 0:
        #         Hypernymy_Group = 'B'
        #     elif setname.find('_IA') > 0:
        #         Hypernymy_Group = 'IA'
        #     elif setname.find('_IB') > 0:
        #         Hypernymy_Group = 'IB'
        #     else:
        #         Hypernymy_Group = 'ALL'
        # else:
        Hypernymy_Group = None

        if EP in ['Hypernymy', 'Location', 'Color', 'Trademark', 'Combined']:  # no templates. Read directly from train/test files
            read_json_examples_fn = dict(Hypernymy=self._read_json_hypernymy_examples,
                              Location=self._read_json_location_examples,
                              Color=self._read_json_color_examples,
                              Trademark=self._read_json_trademark_examples,
                              Combined=self._read_json_combined_examples,
                            )
            filename = os.path.join(dest_dir, setname + '.json')
            # examples = self._read_json_hypernymy_examples(filename) if EP == 'Dir_Hypernymy' else self._read_json_location_examples(filename)
            examples = read_json_examples_fn[EP](filename)
            if is_label_balancing:
                label_balancing = calc_label_balancing(examples)    # for Hypernymy Batch 1: entailment 100%, contradiction: 57%, neutral: 76%)
                examples = balance_labels(examples, label_balancing)
                print("Balanced labels according to dilution ratios: %s"%label_balancing)
            templates = []
            return examples, templates
        else:                       # Generate data from templates and then read it
            dest_dir = templates_dir if dest_dir == '' else dest_dir
            sourcefile = os.path.join(templates_dir, EP + "_" + setname + '.tmpl')
            dest_file =  os.path.join(dest_dir,   EP + "_" + setname + '.txt')
            dest = dest_file if not dest else dest
            templates = self._read_templates(sourcefile)
            self._dump_templates(templates, dest, Numbers_EP_Range, Hypernymy_Group, label_balancing)
            examples = self._create_teaching_examples(self._read_txt(dest))
            # print('%d examples was read from %s...' % (len(examples), dest))
            return examples, templates


    @classmethod
    def _read_templates(cls, template_file):
        """Reads 3 lines value file of the format:
        line0: premis
        line1: hypothesis
        line2: label
        line3..: arguments
                blank line
        ...

        Input: filename
        Output: list of arguments. each template is of the form: (Premise, Hypothesis, Label, (arguments, instances_list))
        """

        def parse_expression(s):
            """ Parse inline expression <<Expression>> e.g. <<3:17:88>>
            Example: "more than <2:17:99> people standing on the street."

            """
            if len(list(re.finditer(':' ,s)) )==2:      # linear numbers list: 4:2:10 ==> 4; 6; 8; 10
                args = s.split(':')
                insts = [str(i) for i in range(int(args[0]), int(args[2] ) +1, int(args[1])  )]
            elif len(list(re.finditer(':' ,s)) )==1:      # linear numbers list: 4:10 ==> 4;5;6;7;8;9;10
                args = s.split(':')
                insts = [str(i) for i in range(int(args[0]), int(args[1] ) +1)]
            return insts


        def parse_instances_str(instances_str, groups = []):
            """
            gets a simple string of list of instances. E.g. "brown; small; cute; amazing; yellow; white"
            no groups inside. only instances or expressions i.e. <<expression>>
            returns a list of instances and their groups. E.g. is the group is ['b1', 'b2']
                instances = [ ('with fur',['b1','b2']),  ('with a small nose',['b1','b2']),   ...]
            :param instances_str:
            :return: instances_n - list of the instances, e.g. ['a', '', 'because of']
            """
            instances_t = instances_str.split(';') # instances_t = ['a', '""', 'dog', '']
            instances_t = [s.strip() for s in instances_t if s != '']   # remove empty instances_str ['a', '""', 'dog', ''] ==> ['a', '""', 'dog']
            instances_t = ['' if (s=='""' or s=="''") else s for s in instances_t]   # ['a', '""', 'dog'] ==> ['a', '', 'dog']

            instances_l = []
            for inst in instances_t:  # looking for and parsing instances expressions (e.g. <<20:2:30>> ==> 20,22,24,...
                start_loc = inst.find('<<')
                if start_loc >= 0:
                    end_loc = inst.find('>>')
                    assert end_loc >0, 'Error in _read_template line %d, file %s: after "<<" must come ">>"' % (i, template_file)
                    expr_list = parse_expression(inst[start_loc + 2:end_loc])
                    instances_l.extend(expr_list)
                else:
                    instances_l.append(inst)
            instances = []
            for inst in instances_l:
                assert not re.search(r'[^a-z/!\-:$\(\)\.\, \'\"0-9]',inst), "invalid character in instance %s line %d, file %s:" % (inst, i, template_file)# only the characters [a-z \'\"0-9] are allowed in instances
                instances.append((inst,groups))  # instances = [ ('with fur',['b1','b2']),  ('with a small nose',['b1','b2']),   ...]
            return instances



        def parse_tepmlate_str(s):
            """ Parse the template line of the form: '*Animal*: dog; cat; snake '
            Input: string s
            Output:
                template: the template string (e.g. 'Animal')
                instances_n: the instances list with their group if exists:  # instances = [ ('with fur',['b1','b2']),  ('with a small nose',['b1','b2']),   ...]

            Example:
                *i*: <a1: John; Max; Nancy; the boy; the kid; the girl>; <a2: They; the boys; the girls; I;>
                *have*: <a1: has; plays with; looks at; points at>;<a2: have; play with; look at; point at>
                *a*: <b1,b3: a>; <b2: "">
                *brown*: brown; small; cute; amazing; yellow; white
                *dog*: <b1: dog; horse; cat;>; <b2: dogs; horses; cats>; <b3: kid; girl; man>
                *with fur*: <b1, b2: with fur; with a small nose; with stripes> ; <b3: ""; with a hat; with a bag>
            """
            def remove_spaces(s_orig):
                # removes all spaces that are NOT between words.
                # E.g. 'b1, b2: with fur; with a small nose   ' ==> 'b1,b2:with fur;with a small nose'
                try:
                    s = s_orig[0] if s_orig[0] != ' ' else ''
                except:
                    print("error in ",s)
                for i in range(1, len(s_orig) - 1):
                    if not (not (s_orig[i - 1] + s_orig[i + 1]).isalnum() and s_orig[i] == ' '):
                        s += s_orig[i]
                s += s_orig[-1] if s_orig[-1] != ' ' else ''
                return s

            loc = s.find('*:', 2)
            template = s[:loc +1]
            instances_str = remove_spaces(s[loc +2:].lower())
            if instances_str[0]=='<':   # if instances_str has groups, i.e instances_str looks like '<b1, b2: with fur; with a small nose; with stripes> ; <b3: ""; with a hat; with a bag>'
                instances_gl = instances_str.split('>;<')
                instances_gl[0] = instances_gl[0][1:]
                if instances_gl[-1][-1] == ';':
                    instances_gl[-1] = instances_gl[-1][:-1]
                assert instances_gl[-1][-1]=='>'    # making sure when deleting the last char, it is '>' as it should be
                instances_gl[-1] = instances_gl[-1][:-1] #instances_gl = ['b1, b2: with fur; with a small nose; with stripes', 'b3: ""; with a hat; with a bag']
                instances = []
                for sg in instances_gl:     # sg = 'b1, b2: with fur; with a small nose; with stripes'
                    loc_g = sg.find(':')
                    assert loc_g>0, 'Error in _read_template lind %d, file %s: after "<" must come "GroupName:"' % (i, template_file)
                    groups = sg[:loc_g].split(',')  # groups = ['b1', 'b3']
                    instances_s = sg[loc_g+1:]            #instances_s = 'with fur; with a small nose; with stripes'
                    instances += parse_instances_str(instances_s, groups) #instances_l = ['with fur', 'with a small nose', 'with stripes']
            else:                               # instances_str = 'brown; small; cute; amazing; yellow; white'
                instances = parse_instances_str(instances_str)   # instances = [ ('with fur',[]),  ('with a small nose',[]),   ...]

            assert s[0] == '*', 'Error in _read_template lind %d, file %s: After Label line the Template must come' % (i, template_file)
            assert loc > -1, 'Error in _read_template lind %d, file %s: template must be of the form *TEMPLATE*:  ... ; ... ;' % (i, template_file)
            return template, instances


        with open(template_file, "r", encoding="utf8", errors='ignore') as f:
            templates, h, l = [], [], []
            ind = 0
            template_found = 0
            content = f.readlines()
            info = {'set':{}, 'template':{}}
            template_title  ={}
            for i, line in enumerate(content + ['\n']):
                s = '' if line == [] else line.lower().strip()

                if s != '' and s[0] == '#':
                    s_cased = line.strip()
                    info['set'] = parse_set_title(s_cased, template_file) if parse_set_title(s_cased, template_file) else info['set']
                    template_title = parse_template_title(s_cased) if parse_template_title(s_cased) else template_title
                    continue
                if ind % 5 == 0 and s == '': continue

                if ind % 5 == 0:  # first line: Premise
                    p = s
                    arguments = []
                if ind % 5 == 1: h.append(s)  # second line: Hypothesis
                if ind % 5 == 2: l.append(s)  # third line: label
                assert (h==[] or l==[]) or (h[-1].find(';')<0 and l[-1].find(';')<0),  'Error in _read_template: found ";" or "<" in line %d, file %s' % (i, template_file)
                if ind % 5 == 3:  # Froth line and more: arguments
                    if s == '' or i == len(content):  # No more arguments for this template:
                        info['template'] = template_title; template_title = {}
                        templates.append((p, h, l, arguments, info.copy()))
                        assert len(h) == len(l), 'Error in _read_template: mismatch number of Hypothesises and Labels. line %d, file %s' % (i, template_file)
                        assert not (template_found == 0 and ((p.find('*') >= 0) or (h[0].find('*') >= 0) or (h[-1].find('*') >= 0))), 'Error in _read_template: "*" found but no template found. line %d, file %s' % (i, template_file)
                        template_found = 0
                        h, l = [], []
                        ind += 1

                    elif s[0]=='*' and s.find('*:')>1 : # template reported
                        try:
                            arguments.append(parse_tepmlate_str(s))
                        except:
                            print("No instantiations in template: %s\nin file %s"%(s,template_file))
                        template_found = 1
                        ind -= 1  # keep looking for more arguments lines
                    else: # another Hypothesis reported
                        h.append(s)
                        ind -= 2 # prepare state to look for label (ind % 5 to be =1)

                ind += 1

            assert ind % 5 == 0, 'Error in _read_template: finished reading the file in the middle of an example. line %d, file %s' % (i, template_file)
            if ind % 5 == 3:  # when there are no arguments, the process needs to add the last example to the list
                templates.append((p, h, l, arguments, info.copy()))
        # print('%d templates was read from %s...' % (len(templates), template_file))
        return templates

    @classmethod
    def _dump_templates(cls, templates, output_file, Numbers_EP_Range = [], Hypernymy_Group = None, label_balancing = None):
        # generates a file with all possible templates from arguments
        # Note! you first must create a matching template file with the same postfix for the automatically generated data files.
        #   For example, to automatically generate create Numbers_templates\EP_datasets\Numbers_C8_2_R400-499.txt you first need to
        #   copy ..\auto_generated_templates\Numbers_C8_2_R60-80.tmpl to Numbers_C8_2_R400-499.tmpl


        MAX_DATASET_SIZE = 20000
        def parse_groups(s_list):
            # converts ['a1, 'a2', 'b1'] to ['a':[1,2], 'b':[1]]
            group_dict = {}
            if s_list == []:
                return group_dict
            for s in s_list:
                name = re.search(r'[A-z]+', s).group()
                assert name != ''
                number = int(re.search(r'[0-9]+', s).group())
                if name in group_dict:
                    group_dict[name].append(number)
                else:
                    group_dict[name] = [number]
            return group_dict

        def is_belongs(arg_groups, template_groups):
            # Checks if the current item belongs to the current instantiation of the group according to its groups (if exist)
            inst_belongs = 1
            for g_name, g_list in arg_groups.items():
                if g_name in template_groups:  # the group has already been seen
                    s_arg = set(arg_groups[g_name])
                    s_tmp = set(template_groups[g_name])
                    if s_arg & s_tmp:  # some elements of the two lists overlap, i.e. valid value
                        template_groups[g_name] = list(s_arg & s_tmp)  # force only the mutual values
                    else:  # if no elements are mutual, i.e. not a valid value
                        inst_belongs = 0
                else:
                    template_groups[g_name] = g_list
            return inst_belongs

        def gen_set_title(set_d):
            s = ''
            if set_d:
                s = "#****** Set Name: %s; Lexical: %s; Syntax: %s; Other: %s *****\n"%(set_d['name'], set_d['lexical'], set_d['syntax'], set_d['other'])
            return s

        def gen_template_title(tmp_d, t_ind):
            if tmp_d:
                return "########### Template: %d; Tense: %s; Syntax: %s; ##############\n"%(t_ind, tmp_d['tense'], tmp_d['syntax'])
            return "################### Template: %d  ###########################\n"%(t_ind)

        def replace_random_for_numbers_EP(p, h, Numbers_EP_Range):
            min, max = Numbers_EP_Range
            span = ut.find_str_after_str(p, '<', 'multiple_words', '>')
            if not span:
                return p, h

            span = '<' + span + '>'
            nums = [random.randint(min, max) for i in range(3)]
            while nums[0]==nums[1] or nums[0]==nums[2] or nums[1]==nums[2]:
                nums = [random.randint(min, max) for i in range(3)]
            low, num, high = sorted(nums)
            p = p.replace('_num_', str(num)).replace('_low_', str(low)).replace('_high_', str(high)).replace('<','').replace('>','')
            h = h.replace('_num_', str(num)).replace('_low_', str(low)).replace('_high_', str(high)).replace('<','').replace('>','')
            return p, h

        def replace_item_class_for_hypernymy(p, h, Hypernymy_Group):
            span = ut.find_str_after_str(p, '<', 'multiple_words', '>')
            if not span:
                return p, h

            span = '<' + span + '>'

            if Hypernymy_Group == 'ALL':        # if 'ALL' groups (task3a) then first randomize any group, and then choose from it.
                Hypernymy_Group = ['A','B'][random.randint(0, 1)]

            if Hypernymy_Group in ['A', 'B']:   # Use different hypernymy classes between the train and test sets (e.g. fruits and vegetable in training, and flowers and trees in test)
                class_rand = random.randint(0,3)
                if Hypernymy_Group=='A':
                    class1 = ['fruits',     'vegetables','tools',   'clothes',  'musical instruments'][class_rand]
                    class2 = ['vegetables', 'fruits',   'clothes',  'tools',    'clothes'            ][class_rand]
                elif Hypernymy_Group=='B':
                    class1 = ['flowers', 'trees',    'mammals', 'insects', 'vehicles'][class_rand]
                    class2 = ['trees',   'flowers',  'insects', 'mammals', 'trees'   ][class_rand]
                item1 = classes[class1][random.randint(0, 9)]
                item2 = classes[class2][random.randint(0, 9)]
            elif Hypernymy_Group in ['IA', 'IB']:   # Use same hypernymy classes in train and test, but use different items for the same classes, so test contains unseen items.
                class_rand = random.randint(0, 3)
                class1 = ['fruits',     'vegetables',   'flowers',    'trees'][class_rand]
                class2 = ['vegetables', 'fruits',       'trees',      'flowers'][class_rand]
                if Hypernymy_Group=='IA':
                    # Take only items from the first half of the classes (items group A = IA)
                    item1 = classes[class1][random.randint(0, 9)]
                    item2 = classes[class2][random.randint(0, 9)]
                elif Hypernymy_Group=='IB':
                    # Take items from classes_rare, which is same class but with rare items that less known by the LM (items group B = IB)
                    item1 = classes_rare[class1][random.randint(0, 7)]
                    item2 = classes_rare[class2][random.randint(0, 7)]
            else:
                raise ValueError("Invalid Hypernymy group")

            p = p.replace('_item_1_', item1).replace('_item_2_', item2).replace('_class_1_', class1).replace('_class_2_', class2).replace('<','').replace('>','')
            h = h.replace('_item_1_', item1).replace('_item_2_', item2).replace('_class_1_', class1).replace('_class_2_', class2).replace('<','').replace('>','')
            return p, h


        def get_hypernymy_items():
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
            return classes, classes_rare

        if os.path.isfile(output_file):
            print("%s already exists. Didn't generate it."%output_file)
            pass
        else:
            # print("generating %s from template..." % output_file)
            with open(output_file, "w") as f:

                # calculating the number of total expected examples

                tot_comb = 0
                for p, h, l, arguments, info in templates:
                    tot_comb_temp = 1
                    for arg, instances in arguments:
                        tot_comb_temp *= len(instances)
                    tot_comb_temp *= len(h)
                    tot_comb += tot_comb_temp
                dilution_ratio = MAX_DATASET_SIZE/tot_comb
                dilution_ratio = min(dilution_ratio,1)
                if dilution_ratio <1:
                    print("Expecting %d examples for %s, so diluting to %1.2f of the originally expected size"%(tot_comb, output_file, dilution_ratio))
                assert label_balancing==None or  (label_balancing['entailment']*dilution_ratio <=1 and label_balancing['contradiction']*dilution_ratio <=1 and label_balancing['neutral']*dilution_ratio <=1), "label_balancing * dilution ratio must be <=1 for all labels"

                counter = 0
                f.write(gen_set_title(templates[0][4]['set']))
                for t_ind, (p, h, l, arguments, info) in enumerate(templates):
                    f.write(gen_template_title(info['template'], t_ind))
                    if arguments == []:  # no template
                        for hi,li in zip(h,l):
                            f.write(p + '\n')
                            f.write(hi + '\n')
                            f.write(li + '\n')
                            f.write('\n')
                            counter += 1
                    else:  # template found
                        for h_ind, (hi, li) in enumerate(zip(h,l)):     # looping over different Hypothesises
                            if len(h)>1:
                                f.write('### Hypothesis: %d\n' % h_ind)
                            f.write('# ' + p + '\n')
                            f.write('# ' + hi + '\n')
                            f.write('# ' + li + '\n')
                            for arg, instances in arguments:
                                f.write('# ' + arg + ': ' + inst2str(instances) + '\n')
                            f.write('\n')
                            ind_vec = [0] * len(arguments)
                            arg_was_used = set()
                            while True:  # scan all possible combinations of instances
                                p_n = p
                                h_n = hi
                                # instanciate all instances instead of the argument variables:
                                template_groups = {}
                                s_args = '# args: '
                                for t, (argument, instances) in enumerate(arguments):  # Run over all arguments: *dog*: ['dog','cat',...], *sleeps*: ['sleeps','jumps',...], ...
                                    p_n_before, h_n_before = p_n, h_n
                                    inst = instances[ind_vec[t]][0].strip()
                                    arg_groups = parse_groups(instances[ind_vec[t]][1]) # converts ['a1, 'a2', 'b1'] to ['a':[1,2], 'b':[1]]
                                    inst_belongs = is_belongs(arg_groups, template_groups) # checks if the instance belongs according to its groups (see example at the beginning of the file)
                                    if inst_belongs==0: break
                                    if Numbers_EP_Range:
                                        p_n, h_n = replace_random_for_numbers_EP(p_n, h_n, Numbers_EP_Range)
                                    if Hypernymy_Group:
                                        classes, classes_rare = get_hypernymy_items()
                                        p_n, h_n = replace_item_class_for_hypernymy(p_n, h_n, Hypernymy_Group)
                                    p_n = p_n.replace(argument, inst)
                                    h_n = h_n.replace(argument, inst)
                                    s_args += " <<%s = %s>> " % (argument, inst)
                                    arg_was_used.add(argument)
                                    assert not (p_n_before == p_n and h_n_before == h_n), 'for argument "%s" there is no instance in premise="%s" and hypothesis="%s" at file %s' % (argument, p, hi, output_file)
                                if not label_balancing:
                                    label_balancing = {'entailment': 1, 'contradiction': 1, 'neutral': 1}
                                if inst_belongs and random.random() < dilution_ratio * label_balancing[li]:     # diluting the dataset size in case there are too many combinations for efficiency
                                    assert p_n.find('*') < 0, 'the premise was not fully instanced: %s' % (p_n)
                                    assert h_n.find('*') < 0, 'the hypothesis was not fully instanced: %s' % (h_n)
                                    f.write(s_args + '\n')
                                    f.write(p_n + '\n')
                                    f.write(h_n + '\n')
                                    f.write(li + '\n')
                                    f.write('\n')
                                    counter +=1
                                ind_vec[-1] += 1
                                for t in range(len(ind_vec) - 1, 0, -1):
                                    if ind_vec[t] == len(arguments[t][1]):
                                        ind_vec[t] = 0
                                        ind_vec[t - 1] += 1
                                if ind_vec[0] == len(arguments[0][1]): break
                            assert len(arg_was_used)==len(arguments), "One of the arguments wasn't used at all"
            print("finished writing %d entailment pairs to %s." % (counter, output_file))

    @classmethod
    def _read_txt(cls, input_file):
        """ Converts NLI text file to a list of Premise-Hypothesis-Label:
        line0: premis
        line1: hypothesis
        line2: label
        line3: blank line
        ...

        Output:
        list of lines: ['The dog is on the floor', 'The animal is on the floor', 'entailment', 'template id = 2',    'There is a boy in the garden', ...]
        lines[0] = Premise
        lines[1] = Hypothesis
        lines[2] = Label
        lines[4]  = Premise
        ...
        """
        with open(input_file, "r") as f:
            content = f.readlines()
            lines = []
            items = []
            template_id, hypothesis_id = '', ''
            info = {'set':{}, 'template':{}, 'hypothesis_id':{}, 'args':{}}
            ind = 0
            for i, line in enumerate(content):
                st = '' if line==[] else line.lower().strip()
                if ind % 4 == 0 and (st == '' or st[0 ]=='#'):
                    s_cased = line.strip()
                    info['set'] = parse_set_title(s_cased) if parse_set_title(s_cased) else info['set']
                    info['template'] = parse_template_title(s_cased) if parse_template_title(s_cased) else info['template']
                    info['args'] = parse_args_title(s_cased)
                    t_loc = st.find ('hypothesis:')           # if there is a template mark at the examples file: "### Template:2 ##### ==> template_id=2
                    if t_loc > 0: info['hypothesis_id'] = st[t_loc +11:t_loc +13].strip()
                    continue
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                if ind %4 == 3:
                    assert (st==''), 'every 4th line must be empty. In line %d at %s' % (i, input_file)
                    items.append(info.copy())       # adding template_id as the last item
                    lines.append(items)
                    items = []
                else:
                    items.append(st)

                ind += 1
            if ind %4 == 3:
                # just in case file ended without enpty line at the end
                lines.append(items)
            assert ind % 4 in [0, 3], 'File ended incorrectly. %s' % (input_file)
            return lines

    def _create_teaching_examples(self, lines):
        """ Converts Lines-list to examples, for the training and dev sets.
        Input - a list of lines from NLI examples file in the order: Premise, Hypothesis, Label, Template ID:
            ['The dog is on the floor', 'The animal is on the floor', 'entailment', 'template id = 2',    'There is a boy in the garden', ...]
        Output - list of examples:
            examples[0].test_a =  'The dog is on the floor'
            examples[0].test_b =  'The animal is on the floor',
            examples[0].label  = 'entailment'
            examples[0].template_id = 'template id = 2'
            ...
        """
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert label in ['entailment', 'contradiction', 'neutral']
            info = line[3]
            guid = 'Template:%s' % info['template']['id']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info.copy()))
        return examples

    # def _read_json_hypernymy_examples(self, filename):
    #     """ Converts json file (generated from mturk) to examples, for the training and dev sets.
    #     Input - json format: Premise, Hypothesis, Label, etc.:
    #     Output - list of examples:
    #         examples[0].test_a =  'The dog is on the floor'
    #         examples[0].test_b =  'The animal is on the floor',
    #         examples[0].label  = 'entailment'
    #         examples[0].info = {"section": "2 hyper->hypo",
    #                             "pword": "operation",
    #                             "hword": "mission",
    #                             "ptype": "hyper",
    #                             "htype": "hypo",
    #                             "row_id": 0,
    #                             "worker_id": "A3J2JEHSHC6T5R",
    #                             "hit_id": "3HYV4299H0XHIIPHHP24PT0LSYAE8M",
    #                             "is_complete": true}
    #     """
    #     examples = []
    #     with open(filename) as json_file:
    #         data = json.load(json_file)
    #     for (i, line) in enumerate(data):
    #         text_a = line['premise']
    #         text_b = line['hypothesis']
    #         label = line['label']
    #         assert label in ['entailment', 'contradiction', 'neutral']
    #         if 'ptype' in line.keys():
    #             info = {
    #                     'row_id':line['row_id'],
    #                     'worker_id':line['worker_id'],
    #                     'is_complete':line['is_complete'],
    #                     'hit_id':line['hit_id'],
    #                     'section':line['section'],
    #                     'ptype':line['ptype'],
    #                     'htype':line['htype'],
    #                     'pword':line['pword'],
    #                     'hword':line['hword'],
    #                     'example_type': 'challenge_set'
    #                     }
    #         else:
    #             info = ''
    #         guid = ''
    #         examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info))
    #     return examples

    def _read_json_location_examples(self, filename):
        """ Converts json file (generated from mturk) to examples, for the training and dev sets.
        Input - json format: Premise, Hypothesis, Label, etc.:
        Output - list of examples:
            examples[0].test_a =  'The dog is on the floor'
            examples[0].test_b =  'The animal is on the floor',
            examples[0].label  = 'entailment'
            examples[0].info = {"section": "1",
                                "row_id": 0,
                                "worker_id": "A3J2JEHSHC6T5R",
                                "hit_id": "3HYV4299H0XHIIPHHP24PT0LSYAE8M",
                                "country": "Portugal",
                                "location": "Porto",
                                "other_location": "London",
                                }
        """
        examples = []
        with open(filename) as json_file:
            data = json.load(json_file)
        for (i, line) in enumerate(data):
            text_a = line['premise']
            text_b = line['hypothesis']
            label = line['label']
            assert label in ['entailment', 'contradiction', 'neutral']
            assert 'country' in line.keys()
            if 'country' in line.keys():
                info = {
                        'row_id':line['row_id'],
                        'worker_id':line['worker_id'],
                        'hit_id':line['hit_id'],
                        'section':line['section'],
                        'country':line['country'],
                        'location':line['location'],
                        'other_location':line['other_location'],
                        'example_type': 'challenge_set'
                        }
            else:
                info = ''
            guid = ''
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info))
        return examples

    def _read_json_color_examples(self, filename):
        """ Converts json file (generated from mturk) to examples, for the training and dev sets.
        Input - json format: Premise, Hypothesis, Label, etc.:
        Output - list of examples:
            examples[0].test_a =  'The dog is on the floor'
            examples[0].test_b =  'The animal is on the floor',
            examples[0].label  = 'entailment'
            examples[0].info = {"section": "1",
                                "row_id": 0,
                                "worker_id": "A3J2JEHSHC6T5R",
                                "hit_id": "3HYV4299H0XHIIPHHP24PT0LSYAE8M",
                                "country": "Portugal",
                                "location": "Porto",
                                "other_location": "London",
                                }
        """
        examples = []
        with open(filename) as json_file:
            data = json.load(json_file)
        for (i, line) in enumerate(data):
            text_a = line['premise']
            text_b = line['hypothesis']
            label = line['label']
            assert label in ['entailment', 'contradiction', 'neutral']
            assert 'color' in line.keys()
            if 'color' in line.keys():
                info = {
                        'row_id': line['row_id'],
                        'worker_id': line['worker_id'],
                        'hit_id': line['hit_id'],
                        'section': line['section'],
                        'color': line['color'],
                        'item': line['item'],
                        'other_color': line['other_color'],
                        'example_type': 'challenge_set'
                        }
            else:
                info = ''
            guid = ''
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info))
        return examples


    def _read_json_trademark_examples(self, filename):
        """ Converts json file (generated from mturk) to examples, for the training and dev sets.
        Input - json format: Premise, Hypothesis, Label, etc.:
        Output - list of examples:
            examples[0].test_a =  'The dog is on the floor'
            examples[0].test_b =  'The animal is on the floor',
            examples[0].label  = 'entailment'
            examples[0].info = {"section": "1",
                                "row_id": 0,
                                "worker_id": "A3J2JEHSHC6T5R",
                                "hit_id": "3HYV4299H0XHIIPHHP24PT0LSYAE8M",
                                "country": "Portugal",
                                "location": "Porto",
                                "other_location": "London",
                                }
        """
        examples = []
        with open(filename) as json_file:
            data = json.load(json_file)
        for (i, line) in enumerate(data):
            text_a = line['premise']
            text_b = line['hypothesis']
            label = line['label']
            assert label in ['entailment', 'contradiction', 'neutral']
            assert 'company' in line.keys()
            if 'company' in line.keys():
                info = {
                        'row_id': line['row_id'],
                        'worker_id': line['worker_id'],
                        'hit_id': line['hit_id'],
                        'section': line['section'],
                        'company': line['company'],
                        'country': line['country'],
                        'country_adj': line['country_adj'],
                        'other_country': line['other_country'],
                        'other_country_adj': line['other_country_adj'],
                        'example_type': 'challenge_set'
                        }
            else:
                info = ''
            guid = ''
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info))
        return examples


    def _read_json_hypernymy_examples(self, filename):
        """ Converts json file (generated from mturk) to examples, for the training and dev sets.
        Input - json format: Premise, Hypothesis, Label, etc.:
        Output - list of examples:
            examples[0].test_a =  'The dog is on the floor'
            examples[0].test_b =  'The animal is on the floor',
            examples[0].label  = 'entailment'
            examples[0].info = {"section": "1",
                                "row_id": 0,
                                "worker_id": "A3J2JEHSHC6T5R",
                                "hit_id": "3HYV4299H0XHIIPHHP24PT0LSYAE8M",
                                "country": "Portugal",
                                "location": "Porto",
                                "other_location": "London",
                                }
        """
        examples = []
        with open(filename) as json_file:
            data = json.load(json_file)
        for (i, line) in enumerate(data):
            text_a = line['premise']
            text_b = line['hypothesis']
            label = line['label']
            assert label in ['entailment', 'contradiction', 'neutral']
            assert 'hypernym' in line.keys()
            if 'hypernym' in line.keys():
                info = {
                        'row_id': line['row_id'],
                        'worker_id': line['worker_id'],
                        'hit_id': line['hit_id'],
                        'section': line['section'],
                        'item': line['item'],
                        'hypernym': line['hypernym'],
                        'hypernym_plural': line['hypernym_plural'],
                        'other_hypernym': line['other_hypernym'],
                        'other_hypernym_plural': line['other_hypernym_plural'],
                        'example_type': 'challenge_set'
                        }
            else:
                info = ''
            guid = ''
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info))
        return examples


    def _read_json_combined_examples(self, filename):
        """ Converts json file (generated from mturk) to examples, for the training and dev sets.
        Input - json format: Premise, Hypothesis, Label, etc.:
        Output - list of examples:
            examples[0].test_a =  'The dog is on the floor'
            examples[0].test_b =  'The animal is on the floor',
            examples[0].label  = 'entailment'
            examples[0].info = {"section": "1",
                                "row_id": 0,
                                "worker_id": "A3J2JEHSHC6T5R",
                                "hit_id": "3HYV4299H0XHIIPHHP24PT0LSYAE8M",
                                "country": "Portugal",
                                "location": "Porto",
                                "other_location": "London",
                                }
        """
        examples = []
        with open(filename) as json_file:
            data = json.load(json_file)
        for (i, line) in enumerate(data):
            text_a = line['premise']
            text_b = line['hypothesis']
            label = line['label']
            assert label in ['entailment', 'contradiction', 'neutral']
            assert ('hypernym' in line.keys()) or ('company' in line.keys()) or ('color' in line.keys()) or ('location' in line.keys())
            if 'location' in line.keys():
                info = {
                        'row_id':line['row_id'],
                        'worker_id':line['worker_id'],
                        'hit_id':line['hit_id'],
                        'section':line['section'],
                        'country':line['country'],
                        'location':line['location'],
                        'other_location':line['other_location'],
                        'example_type': 'challenge_set'
                        }
            if 'color' in line.keys():
                info = {
                        'row_id': line['row_id'],
                        'worker_id': line['worker_id'],
                        'hit_id': line['hit_id'],
                        'section': line['section'],
                        'color': line['color'],
                        'item': line['item'],
                        'other_color': line['other_color'],
                        'example_type': 'challenge_set'
                        }
            if 'company' in line.keys():
                info = {
                        'row_id': line['row_id'],
                        'worker_id': line['worker_id'],
                        'hit_id': line['hit_id'],
                        'section': line['section'],
                        'company': line['company'],
                        'country': line['country'],
                        'country_adj': line['country_adj'],
                        'other_country': line['other_country'],
                        'other_country_adj': line['other_country_adj'],
                        'example_type': 'challenge_set'
                        }
            if 'hypernym' in line.keys():
                info = {
                        'row_id': line['row_id'],
                        'worker_id': line['worker_id'],
                        'hit_id': line['hit_id'],
                        'section': line['section'],
                        'item': line['item'],
                        'hypernym': line['hypernym'],
                        'hypernym_plural': line['hypernym_plural'],
                        'other_hypernym': line['other_hypernym'],
                        'other_hypernym_plural': line['other_hypernym_plural'],
                        'example_type': 'challenge_set'
                        }
            guid = ''
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, info=info))
        return examples

if __name__ == "__main__":
    # templates_dir = '../teaching_data/Numbers_Templates/auto_generated_templates'
    templates_dir = '../teaching_data/Hypernymy_Templates/auto_generated_templates'
    json_file = 'hypernymy_dataset1.json'
    processor = TemplateProcessor()
    # examples = processor._read_json_hypernymy_examples(json_file)
    # pass
    # exit(0)

    # train_examples, train_templates = processor.get_templates(templates_dir, 'Numbers', 'C8_test')
    # train_examples, train_templates = processor.get_templates(templates_dir, 'Datives', 'T1_')
    train_examples, train_templates = processor.get_templates(templates_dir, 'Hypernymy', 'S0_2_B', is_label_balancing = True)
    train_examples, train_templates = processor.get_templates(templates_dir, 'Hypernymy', 'C1_1_A', is_label_balancing = True)
    print("end")
