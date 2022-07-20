import numpy as np
import json
import math
import string
import nltk

import stanza

from gensim import models
from gensim.models import Word2Vec
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from unidecode import unidecode

from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import*
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import math
import torch
from flair.embeddings import FlairEmbeddings


from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

import sys
import time
import os

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.worlds import create_task,validate

def pre_process(corpus):
    # convert input corpus to lower case.
    corpus = corpus.lower()
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations from string.
    # word_tokenize is used to tokenize the input corpus in word tokens.
    corpus = " ".join([i for i in word_tokenize(corpus) ])
    #corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    corpus = unidecode(corpus)
    return corpus

def get_tokens(text):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def lemmatize_content_word(path_in, path_out):
    document = []
    tag = {}
    dict_dialogue = {}
    lmtzr = WordNetLemmatizer()
    
    dialogs = []
    check = path_in.split('.')[-1]
    if check == 'txt':
        with open(path_in,'r') as load_f:
            for line in load_f:
                turn = line.split(': ', 1 )
                turn[1] = turn[1].replace('\n', '')
                dialogs.append(turn)
    for turn in dialogs:
        document.append(turn[1])
    
    for turn in document:
        tokens = get_tokens(turn)
        #lemmatized_tokens = lmtzr.lemmatize(word)
        stemmer = PorterStemmer()
        stemmed = stem_tokens(tokens, stemmer)
        for i in range(len(tokens)):
            result = nltk.pos_tag(tokens)
            tag[result[i][0]] = result[i][1]
            if (tag[tokens[i]] == 'FW') or (tag[tokens[i]] == 'JJ') or (tag[tokens[i]] == 'NN') or (tag[tokens[i]] == 'NNS') or (tag[tokens[i]] == 'NNPS') or (tag[tokens[i]] == 'PDT') or (tag[tokens[i]] == 'POS') or (tag[tokens[i]] == 'VB') or (tag[tokens[i]] == 'VBD') or (tag[tokens[i]] == 'VBG') or (tag[tokens[i]] == 'VBN') or (tag[tokens[i]] == 'VBP') or (tag[tokens[i]] == 'VBZ') : 
                dict_dialogue[tokens[i]] = stemmed[i]
                
        pop_list = []
        for k,v in dict_dialogue.items():
            if (k == 's') or (k == 'm') or (k == 're') or (k == 'ont') or (k == 'll') or (k == 've') or (k == 'ive') or (k == 't') or (k == 'd') :
                pop_list.append(k)
        for pop_it in pop_list:   
            dict_dialogue.pop(pop_it)
    with open(path_out, 'w') as f:
        json.dump(dict_dialogue, f)
    #return dict_dialogue
    

    
def tf(word, count):
    return count[word] / sum(count.values())
def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)
def idf(word, count_list):
    return math.log(len(count_list)) / (1 + n_containing(word, count_list))
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)
def count_term(text):
    stemmed = []
    
    tokens = get_tokens(text)
    stemmer = PorterStemmer()
    stemmed = stem_tokens(tokens, stemmer)
    
    count = Counter(stemmed)
    return count


def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def sorted_words_idf(path_log, path_dict_lemma):
    with open(path_dict_lemma, 'r') as f:
        dict_dialogue = json.load(fp=f)
    document = []
    dialogs = []
    with open(path_log,'r') as load_f:
        for line in load_f:
            turn = line.split(': ', 1 )
            turn[1] = turn[1].replace('\n', '')
            dialogs.append(turn)
    for turn in dialogs:
        document.append(turn[1])
    countlist = []
    sorted_idf = {}
    for text in document:
        countlist.append(count_term(text))
    for k,v in dict_dialogue.items():
        sorted_idf[v] = idf(v,countlist)
    sorted_idf = sorted(sorted_idf.items(),key = lambda item:item[1],reverse=True)
    idf_list = [item[0] for item in sorted_idf[:int(len(sorted_idf))]]
    return idf_list



def normalize(text):
    return stem_tokens(get_tokens(text),PorterStemmer())

def get_tfidf_cosine(document):
    vect = TfidfVectorizer(tokenizer=normalize, min_df=1)                                                                                                                                                                                                   
    tfidf = vect.fit_transform(document) 
    pairwise_similarity = (tfidf * tfidf.T).toarray()
    #print(pairwise_similarity.shape)
    return pairwise_similarity





def get_w2v_feature_vectors(document,i,j,w):
    
    #w.init_sims(replace=True)
    toks1 = [token for token in document[i].split() if token in w.vocab]
    vecs1 =[w[tok] for tok in toks1] 
    toks2 = [token for token in document[j].split() if token in w.vocab]
    vecs2 =[w[tok] for tok in toks2]
    res = []
    for vec in [vecs1, vecs2]:
        if vec == [] :
            res.append(None)
        else :
            res.append(np.mean(vec,axis=0))
    return res




def get_cosine_w2v(document,i, j, w):
    
    feature_vectors = get_w2v_feature_vectors(document, i,j, w)
    
    feature_vec_1 = feature_vectors[0]
    feature_vec_2 = feature_vectors[1]    
    
    if np.shape(feature_vec_1) == () or np.shape(feature_vec_2) == ():
        res = 0
    else :
        res = cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
    return res


def jaccard_similarity(document,i,j):
    text1 = document[i]
    text2 = document[j]
    list1 = normalize(text1)
    list2 = normalize(text2)
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union != 0 :
        res = float(intersection) / union
    else :
        res = 0
    return res



class GameEvaluation(object):


    def __init__(self, path, path_dict, simfunc, rep, contradict, bonus, pro,flu, spec, nli, know, model):
        self.path = path  
        self.simfunc = simfunc
        self.contradict = contradict
        self.rep = rep
        self.path_dict = path_dict
        self.bonus = bonus
        self.model = model
        self.pro = pro
        self.flu = flu
        self.spec = spec

        self.nli = nli
        self.know = know
        
    def entity_count(self):
        nlp = stanza.Pipeline('en')
        f = open(self.path)
        lines = f.readlines()
        entites = {lines[0].split(':')[0]:[],lines[1].split(':')[0]:[]}
        names = []
        index = 0
        for line in lines:
            res = nlp(line.split(':')[1]).entities
            if res !=[]:
                if index%2 == 0:
                    for ent in res:
                        if ent.type !=  'DATE' and ent.type !=  'TIME' and ent.type !=  'ORDINAL' and ent.type !=  'CARDINAL' and ent.text not in names:
                            names.append(ent.text)
                            entites[lines[0].split(':')[0]].append(ent)
                else:
                    for ent in res:
                        if ent.type !=  'DATE' and ent.type !=  'TIME' and ent.type !=  'ORDINAL'and ent.type !=  'CARDINAL' and ent.text not in names:
                            names.append(ent.text)
                            entites[lines[1].split(':')[0]].append(ent)
            index += 1
        count_entities = {}
        for k,v in entites.items():
            count_entities[k] = len(v)
        
        return count_entities  
       
    
    def get_dialogue(self):
        file_path = self.path
        dialogs = []
        with open(file_path,'r') as load_f:
            for line in load_f:
                turn = line.split(': ', 1 )
                turn[1] = turn[1].replace('\n', '')
                dialogs.append(turn)
        return dialogs
    
    def get_dialogue_turns(self):
        file_path = self.path
        dialogs = []
        with open(file_path,'r') as load_f:
            for line in load_f:
                turn = line.split(': ', 1 )
                dialogs.append(turn[1].replace('\n', ''))
        return dialogs
    
    def t(self,i):
        dialogs = self.get_dialogue()
        return dialogs[i][1]

    def H(self,i):
        his = []
        speaker = self.get_dialogue()[i][0]
        k = 0
        for turn in self.get_dialogue():
            if turn[0] == speaker and k<i:
                his.append([k,turn[1]])
            k = k+1
        return his
    
    def get_tfidf_sim(self):
        document = self.get_dialogue_turns()
        return get_tfidf_cosine(document)
    
    def get_cosine_sim(self,i,j,mat_tf):
        document = self.get_dialogue_turns()
        if self.simfunc == 'w2v':
            res =  get_cosine_w2v(document,i, j, self.model)
        elif self.simfunc == 'jaccard':
            res = jaccard_similarity(document,i,j)
        elif self.simfunc == 'tfidf':
            res = mat_tf[i,j]
        return res
    
    
        
    
    def get_nli_score(self):
        res = []
        nli_opt = Opt()
        nli_file = "/tmp/BERT_dnli.opt"
        nli_opt = nli_opt.load(nli_file)
        nli_opt['interactive_mode'] = nli_opt.get('interactive_mode', True)
        #nli_opt['sep_last_utt'] = nli_opt.get('sep_last_utt', True)
        agent_nli = create_agent(nli_opt)
        
        dialogs = self.get_dialogue()
        num_turns = len(dialogs)
        contradict_point = [0]*num_turns
        for i in range(num_turns):
            self_history = []
            j = i-2
            while j >=0 :
                self_history.append(self.t(j))
                j = j-2
            contradict_point[i] = 0
            for sent in self_history: 
                context = "Premise: "+sent+"\n"+"Hypothesis: "+ self.t(i)
                agent_nli.observe({'text': context, 'episode_done': True})
                response_nli = agent_nli.act()
                if response_nli == "contradiction":
                    contradict_point[i] += 1
        point_0 = 0
        point_1 = 0
        ind_0 = 0
        ind_1 = 1
        while ind_0 < num_turns:
            point_0 += contradict_point[ind_0]
            ind_0 = ind_0 + 2
        while ind_1 < num_turns:
            point_1 += contradict_point[ind_1]
            ind_1 = ind_1 + 2
        
        res.append(point_0)
        res.append(point_1)
        
        return res
    
    
    
    def prev(self,i):
        return self.t(i-1)
    def get_points(self):
        #initialization
        
        if self.flu != 0:
            language_model = FlairEmbeddings('news-forward').lm
        
        
        
        dialogs = self.get_dialogue()
        num_turns = len(dialogs)
        Contradict = np.zeros((num_turns,))
        Repeat = np.zeros((num_turns,))
        Fluency = np.zeros((num_turns,))
        Proact = np.zeros((num_turns,))
        Know = np.zeros((num_turns,))
        Ent = self.entity_count()
        
        Rep_q = [[] for i in range(num_turns)]
        Rep_u = [[] for i in range(num_turns)]
        last_pos = {}
        bonus = 0
        list_words=[]
        mat_tf = self.get_tfidf_sim()
        
        Rep_index = set()
        Div1 = []
        Div2 = []
        TEXT_1 = []
        TEXT_2 = []
        complete_dial = self.get_dialogue_turns()
        
        
        
        with open(self.path_dict,'r') as load_f:
            dict_dialogue = json.load(load_f)
        
        idf_list = sorted_words_idf(self.path, self.path_dict)
        lim = 1 
        idf_per = 0.5 
        BONUS = [False]
        for i in range(num_turns)[1:] :
           
            current_turn = self.t(i)
            if "don't know" in current_turn:
                Know[i] += 1
            
            if '?' in current_turn:
                Proact[i] = Proact[i] + 1
            if current_turn == '':
                Repeat[i] = Repeat[i] + 1
             
            get_bonus = False
            for word in current_turn.split() :
                
                if (word in dict_dialogue.keys()) and idf_list.index(dict_dialogue[word]) < len(idf_list) * idf_per:
                    if dict_dialogue[word] in last_pos.keys():
                        if i - last_pos[dict_dialogue[word]] > lim:
                            get_bonus = True
                            list_words.append((i,word))
                    last_pos[dict_dialogue[word]] = i
            
            BONUS.append(get_bonus)
            if i < 3:
                continue
            
            #detect inconsistency
            if Rep_q[i-1] != []:
                rep_t = []
                flag_inc = 0
                for rep in Rep_q[i-1]:
                    rep_t.append(rep[0])
                for rep_index in rep_t:
                    consis_turn = self.t(rep_index+1)
                    if self.get_cosine_sim(i,rep_index+1,mat_tf) < 0.8:
                        flag_inc = flag_inc+1
                if flag_inc != 0:
                    Contradict[i] = Contradict[i] +1
            #detect repetition
            for his_turn in self.H(i):
                if self.get_cosine_sim(i,his_turn[0],mat_tf) >= 0.95:
                    Rep_u[i].append(his_turn)
            if len(Rep_u[i]) > 0:
                flag_qr = 0
                for rep_u in Rep_u[i]:
                    if rep_u[1].split()[-1] == '?' and current_turn.split()[-1] == '?':
                        flag_qr = flag_qr + 1
                        Rep_q[i].append(rep_u)
                if flag_qr > 0:
                    Repeat[i] = Repeat[i] + 1
                    Rep_index.add(i)
                else:
                    if Rep_q[i-1] == [] :
                        Repeat[i] = Repeat[i] + 1
                        Rep_index.add(i)
                    else :
                        flag_rep = 0
                        for check_index in Rep_q[i-1] :
                            rep_turn = self.t(rep_index+1)
                            if self.get_cosine_sim(i,rep_index+1,mat_tf) <=0.95:
                                flag_rep = 0
                        if flag_rep > 0:
                            Repeat[i] = Repeat[i] + 1
                            Rep_index.add(i)
            
        
            
            
            if len(current_turn) <= 1:
                Fluency[i] = 50
              
            else:
                Fluency[i] = min(50,language_model.calculate_perplexity(current_turn))
                
        for i in range(len(complete_dial)):
            if i not in Rep_index:
                if i%2 == 0:
                    TEXT_1.append(complete_dial[i])
                else:
                    TEXT_2.append(complete_dial[i])
        distinct_1 = distinct(TEXT_1)
        distinct_2 = distinct(TEXT_2)
        Div1.append(distinct_1[0])
        Div1.append(distinct_2[0])
        Div2.append(distinct_1[1])
        Div2.append(distinct_2[1])
            
            
        #return [Repeat, Contradict, BONUS, Fluency, Proact, Div1, Div2, Know, list_words,Rep_index]
        return [Repeat, Contradict, BONUS, Fluency, Proact, Div1, Div2, Ent, list_words,Rep_index]
    def count_numbers(self):
        res = self.get_points()
        R = res[0]
        C = res[1]
        B = res[2]
        F = res[3]
        P = res[4]
        DIV_1 = res[5]
        DIV_2 = res[6]
        K = res[7]
        Rep_index = res[9]
        
        result = [sum([F[i] for i in range(len(F))])/len(F),sum([K[i] for i in range(len(R))]),sum([P[i] for i in range(len(R))]),(res[5][0]+res[6][0]+res[6][1]+res[5][1])/4,sum([R[i] for i in range(len(R))]),sum([C[i] for i in range(len(R))]),sum([B[i] for i in range(len(R))]),len(F)]
        return result
        
    def count_fluency(self):
        res = self.get_points()
        R = res[0]
        C = res[1]
        B = res[2]
        F = res[3]
        P = res[4]
        DIV_1 = res[5]
        DIV_2 = res[6]
        K = res[7]
        
        L_1 = []
        L_2 = []
        
        max_l1 = 0
        max_l2 = 0
        max_id1 = 0
        max_id2 = 0
        
        Rep_index = res[9]
        
        dialogue = self.get_dialogue()
        id_1 = dialogue[1][0]
        id_2 = dialogue[2][0]
        
        for i in range(int((len(R)-1)/2)):
            if 2*i+1 not in Rep_index:
                L_1.append(2*i+1)
                if max_l1 < F[2*i+1]:
                    max_l1 = F[2*i+1]
                    max_id1 = 2*i+1
                
                 
                
        for i in range(int((len(R)-1)/2)):
            if 2*i+2 not in Rep_index:
                L_2.append(2*i+2)
                if max_l2 < F[2*i+2]:
                    max_l2 = F[2*i+2]
                    max_id2 = 2*i+2
        
        result_R = [sum([R[2*i+1] for i in range(int((len(R)-1)/2))]),sum([R[2*i+2] for i in range(int((len(R)-1)/2))])]
        #result_K = [sum([K[i] for i in L_1]),sum([K[i] for i in L_2])]
        result_K = [K[id_1],K[id_2]]
        result_P = [sum([P[i] for i in L_1]),sum([P[i] for i in L_2])]       
        result_B = [sum([B[i] for i in L_1]),sum([B[i] for i in L_2])]       
        result_C = [sum([C[i] for i in L_1]),sum([C[i] for i in L_2])]
        if len(L_1) == 1 and len(L_2) == 1:
            
            result_F = [sum([F[i] for i in L_1]),sum([F[i] for i in L_2])]
        elif len(L_1) == 1:
            result_F = [sum([F[i] for i in L_1]),(sum([F[i] for i in L_2])-max_l2)/(len(L_2)-1)]
        elif len(L_2) == 1:
            result_F = [sum([F[i] for i in L_2]),(sum([F[i] for i in L_1])-max_l1)/(len(L_1)-1)]
        else:
            result_F = [(sum([F[i] for i in L_1])-max_l1)/(len(L_1)-1),(sum([F[i] for i in L_2])-max_l2)/(len(L_2)-1)]
        #result = [max_l1,max_id1,dialogue[max_id1],max_l2,max_id2,dialogue[max_id2]]
        result_D = [(DIV_1[0]+DIV_2[0])/2,(DIV_1[1]+DIV_2[1])/2]
        S_1 = int(result_R[0]<result_R[1]) + int(result_K[0]>result_K[1]) + int(result_P[0]>result_P[1]) + int(result_B[0]>result_B[1]) + int(result_C[0] < result_C[1]) + int(result_F[0]<result_F[1])+int(result_D[0]>result_D[1])
        
        S_2 = int(result_R[0]>result_R[1]) + int(result_K[0]<result_K[1]) + int(result_P[0]<result_P[1]) + int(result_B[0]<result_B[1]) + int(result_C[0] > result_C[1]) + int(result_F[0]>result_F[1])+int(result_D[0]<result_D[1])
        
        return [[id_1,S_1],[id_2,S_2]]
        
        
        
        
        
    def Scoring_for_bot(self):
        res = self.get_points()
        nli_res = [0]*2
        if self.nli != 0:
            nli_res = self.get_nli_score()
        R = res[0]
        C = res[1]
        B = res[2]
        F = res[3]
        P = res[4]
        DIV_1 = res[5]
        DIV_2 = res[6]
        K = res[7]
        #E = res[7]
        L_1 = []
        L_2 = []
        Rep_index = res[9]
        for i in range(int((len(R)-1)/2)):
            if i not in Rep_index:
                L_1.append(2*i+1)
        for i in range(int((len(R)-1)/2)):
            if i not in Rep_index:
                L_2.append(2*i+2)
        
    
        
        dialogue = self.get_dialogue()
        id_1 = dialogue[1][0]
        id_2 = dialogue[2][0]
        
        R_1 = int(sum([R[2*i+1] for i in range(int((len(R)-1)/2))]) < sum([R[2*i+2] for i in range(int((len(R)-1)/2))]))
        R_2 = int(sum([R[2*i+1] for i in range(int((len(R)-1)/2))]) > sum([R[2*i+2] for i in range(int((len(R)-1)/2))]))
        
        C_1 = int(sum([C[2*i+1] for i in range(int((len(R)-1)/2))]) < sum([C[2*i+2] for i in range(int((len(R)-1)/2))]))
        C_2 = int(sum([C[2*i+1] for i in range(int((len(R)-1)/2))]) > sum([C[2*i+2] for i in range(int((len(R)-1)/2))]))
        
        B_1 = int(sum([B[2*i+1] for i in range(int((len(R)-1)/2))]) > sum([B[2*i+2] for i in range(int((len(R)-1)/2))]))
        B_2 = int(sum([B[2*i+1] for i in range(int((len(R)-1)/2))]) < sum([B[2*i+2] for i in range(int((len(R)-1)/2))]))
        
        
        F_1 = int(sum([F[i] for i in L_1])/len(L_1) < sum([F[i] for i in L_2])/len(L_2))
        F_2 = int(sum([F[i] for i in L_1])/len(L_1) > sum([F[i] for i in L_2])/len(L_2))
        
        P_1 = int(sum([P[2*i+1] for i in range(int((len(R)-1)/2))]) > sum([P[2*i+2] for i in range(int((len(R)-1)/2))]))
        P_2 = int(sum([P[2*i+1] for i in range(int((len(R)-1)/2))]) < sum([P[2*i+2] for i in range(int((len(R)-1)/2))]))
        
        Spec_1 = int((res[5][0]+res[6][0])/2 > (res[5][1]+res[6][1])/2)
        Spec_2 = int((res[5][0]+res[6][0])/2 < (res[5][1]+res[6][1])/2)
        
        
        K_1 = int(sum([K[2*i+1] for i in range(int((len(R)-1)/2))]) < sum([K[2*i+2] for i in range(int((len(R)-1)/2))]))
        K_2 = int(sum([K[2*i+1] for i in range(int((len(R)-1)/2))]) > sum([K[2*i+2] for i in range(int((len(R)-1)/2))]))
        
        
        
        S_1 = R_1 + C_1 + B_1 + F_1 + P_1 +  Spec_1  +K_1 #+ E_1
        S_2 = R_2 + C_2 + B_2 + F_2 + P_2 +  Spec_2  +K_2 #+ E_2
        

        
       
        
        return [[id_1,S_1],[id_2,S_2]]

    
    
def main(argv):
    path = argv[1]
    path_dict = argv[2]
    simfunc = argv[3]
    
    
    contradict = argv[5]
    rep = argv[4]
    bonus = argv[6]
    pro = argv[7]
    flu = argv[8]
    spec = argv[9]
    nli = int(argv[10])
    know = int(argv[11])
    
    if simfunc == 'w2v' :
        m = models.KeyedVectors.load_word2vec_format('/root/GoogleNews-vectors-negative300.bin', binary=True)
    else :
        m = None
    
    T1 = time.time()
    game = GameEvaluation(path, path_dict, simfunc, rep,contradict,bonus,pro,flu,spec,nli,know,model = m)
    
    T2 = time.time()
    
    
    
    
    
    
    

if __name__ == "__main__":
    main(sys.argv)    




