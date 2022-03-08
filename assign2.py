import nltk
from nltk.corpus import brown
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from collections import Counter
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
import pickle
from os import linesep
import json
import collections
import re
from collections import OrderedDict
from operator import itemgetter
import pytrec_eval

def train_model(n, corpus):

  # corpus_pp = list(flatten(pad_both_ends(sent, n) for sent in corpus_processed))
  train_, vocab_ = padded_everygram_pipeline(n, corpus)


    if(n==1):
        lm = Laplace(n)
        lm.fit(train_, vocab_)
    else:
        lm = KneserNeyInterpolated(n)
        # lm = Laplace(n)
        lm.fit(train_, vocab_)

    save_model(lm, n)
    return lm
  

def save_model(lm, n):
    with open('%s_gram_model.pkl' % n, 'wb') as fout:
        pickle.dump(lm, fout)
    return
    

# load birbeck corpus
def load_birbeck_corpus():

  birbeck_list = dict()
  f = open("/content/APPLING1DAT.643", "r")
  content = f.readlines()

  for line in content:
    if(line.startswith('$')):
        continue
    
    pair = line.strip().split()
    correct_word = pair[1].strip()
    sentence = ' '.join(pair[2:]).split('*')[0].strip()

    # print(sentence, " : ", correct_word)
    birbeck_list[sentence] = correct_word
  return birbeck_list
 
 

def score(n, vocabulary, birbeck_list):

  with open('%s_gram_model.pkl' % n, 'rb') as fin:
    lm = pickle.load(fin)

  n_prob = collections.defaultdict(dict)
  for line in birbeck_list.keys():
    for word in vocabulary:
        
      temp = line.split()[-(n-1):]
      if(len(temp) < n-1 ):
        for i in range(n-len(temp)-1):
            temp.insert(0,'<s>')

      score = lm.score(word, temp)
      n_prob[line][word] = score
    n_prob[line] = dict(sorted(n_prob[line].items(), key=lambda item: item[1], reverse = True)[:10])


  with open('score_%s_gram.json' % n, 'w') as fp:
    json.dump(n_prob, fp, indent=2)
  return n_prob
  
  
def clean_dict(dictionary):
  len(dictionary)
  vocab = []
  regexp = re.compile('^[^a-zA-Z]+')
  for word in dictionary:
    # if(word.startswith("[a-zA-Z]")):
    if ( regexp.search(word) or len(word) == 1 ):
      continue
    vocab.append(word.lower())

  return vocab
  
  
  

def sorted_dict_elements_evaluate(correct_dict, prob_n_gram, n):
  
    sort_dict_10 = collections.defaultdict(dict)
    sort_dict_5 = collections.defaultdict(dict)
    sort_dict_1 = collections.defaultdict(dict)

    for i in prob_n_gram:
        sort_dict_10[i] =  (OrderedDict(sorted( prob_n_gram[i].items(), key=itemgetter(1))[:10]))
        sort_dict_5[i] =  (OrderedDict(sorted( prob_n_gram[i].items(), key=itemgetter(1))[:5]))
        sort_dict_1[i] =  (OrderedDict(sorted( prob_n_gram[i].items(), key=itemgetter(1))[:1]))
    with open('sort_dict_%s.json' % n, 'w') as fp:
        json.dump(sort_dict_10, fp, indent=2)
        json.dump(sort_dict_5, fp, indent=2)
        json.dump(sort_dict_1, fp, indent=2)

    evaluate(correct_dict, sort_dict_10, sort_dict_5, sort_dict_1, n )



def calculate_correct_dict(birbeck_list):
  correct_dict = collections.defaultdict(dict)
  for item in birbeck_list.keys():
    correct_dict[item] = {birbeck_list[item] : 1}
  return correct_dict
  
  
  

def evaluate(correct_dict, sort_dict_10, sort_dict_5, sort_dict_1, n ):
  
  results = collections.defaultdict(dict)

  for item in correct_dict:
      results[item] = {}
      
      if(list(correct_dict[item].keys())[0] in sort_dict_1[item].keys()):
          results[item][list(correct_dict[item])[0]] = 1
      
      for k in list(sort_dict_5[item].keys()):
          if( k not in results[item].keys()):
              results[item][k] = 1/5
              
      for k in list(sort_dict_10[item].keys()):
          if( k not in results[item].keys()):
              results[item][k] = 1/10  

  evaluator = pytrec_eval.RelevanceEvaluator(correct_dict, {'success'})
  res = evaluator.evaluate(results)
  
  print("Results for {}-gram-model: ".format(n))
  for measure in sorted(list(res[list(res.keys())[0]].keys())):
        print('average', measure, ': ', pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in res.values()]))




def train_calculate(n, corpus, vocabulary, birbeck_list):

  train_model(n, corpus)
  prob_n_gram = score(n, vocabulary, birbeck_list)

  return prob_n_gram
  
  
if __name__ == "__main__":

    nltk.download('brown')
    nltk.download('punkt')


    tokens = brown.words(categories='news')
    vocabulary = clean_dict(set(tokens))

    birbeck_list = load_birbeck_corpus()

    corpus = brown.sents(categories='news')
    correct_dict = calculate_correct_dict(birbeck_list)

    n = 1
    prob_n_gram = train_calculate(n, corpus, vocabulary, birbeck_list)
    sorted_dict_elements_evaluate(correct_dict, prob_n_gram, n)

    n = 2
    prob_n_gram = train_calculate(n, corpus, vocabulary, birbeck_list)
    sorted_dict_elements_evaluate(correct_dict, prob_n_gram, n)

    n = 3
    prob_n_gram = train_calculate(n, corpus, vocabulary, birbeck_list)
    sorted_dict_elements_evaluate(correct_dict, prob_n_gram, n)

    n = 5
    prob_n_gram = train_calculate(n, corpus, vocabulary, birbeck_list)
    sorted_dict_elements_evaluate(correct_dict, prob_n_gram, n)

    n = 10
    prob_n_gram = train_calculate(n, corpus, vocabulary, birbeck_list)
    sorted_dict_elements_evaluate(correct_dict, prob_n_gram, n)
