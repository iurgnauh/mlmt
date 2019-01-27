import io
from collections import defaultdict
import sys
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def translate(file_path,Translation,src_id2word,tgt_id2word):
    #buffer size?
    f_output = io.open('better_translation/en-es-cheap_train.txt', 'w', encoding='utf-8')

    in_lines = io.open(file_path,'r',encoding='utf8').readlines()
    index = 0
    
    word2id = {v: k for k, v in src_id2word.items()}
    for line in in_lines[1:]:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            label = pairs[-1]
            
            #upper case and lower case do matters
            #looks like the fasttext english embedding only have lower case
            #how about the mlmt code?
            
            if word in word2id or word.lower() in word2id:
                #get the index of neibors
                if word in word2id:
                    word_index = word2id[word]
                else:
                    word_index = word2id[word.lower()]
                neibors_index =  Translation[word_index]
                for idx in neibors_index:
                    f_output.write(tgt_id2word[idx]+u" "+label+'\n')
                    #print(ele)
            else:
                #otherwise, just copy the word from source language
                f_output.write(word+u" "+label+'\n')
                #print ("out of vocab word: ",word)
            index = index + 1
        elif len(line) < 2:
            f_output.write(u" "+'\n')
            #You can also force flush the buffer to a file programmatically with the flush() method.
            f_output.flush()
        if index % 1000==0:
            print (index)
        #to do add sentence spliter here:
    f_output.close()

def counters():
    return defaultdict(int)

def freqs(LofD):
    r = defaultdict(counters)
    for d in LofD:
        for k, v in d.items():
            r[k][v] += 1
    return dict((k, dict(v)) for k, v in r.items())

def sentence_score(score_table,temp_words,temp_labels):
    score = 0
    #get the score for the sentence
    num = 0 #entity num
    for (word,label) in zip(temp_words,temp_labels):
        if label !='O':
            score+= score_table[word][label]
            num +=1
    if num==0:
        return 0,num
    else:
        return float(score)/num, num
    
def generate_frequency_table(trans_path,output_path):
    f_output = io.open(output_path, 'w', encoding='utf-8')
    in_lines = io.open(trans_path,'r',encoding='utf8').readlines()
    frequency = 0
    total_sentence = 0
    count = 0
    temp_words = []
    temp_labels = [] 
    #the frequency table
    entity_table = defaultdict(counters)
    
    
    #generate the required table
    #score_table = dict((k, dict(v)) for k, v in entity_table.items())
    
    for line in in_lines:
        
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            label = pairs[-1]
            if label!='O':
                #print(label)
                #no need to record any other not entity?
                entity_table[word][label]+=1
    
    #get the frequency table we need
    score_table = dict()
    for k,v  in entity_table.items():
        new_v = {key:float(value)/sum(v.values()) for (key,value) in v.items()}
        score_table[k] = new_v
        
    
    #in_lines = io.open(trans_path,'r',encoding='utf8').readlines()
    
    for line in in_lines:
        
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            label = pairs[-1]
            temp_words.append(word)
            temp_labels.append(label)
            
            
            
            if label!='O':
                #print(label)
                count = count + 1
                
        else:
            #print("here")
            #print (count)
            total_sentence = total_sentence+1
            #may change the data filter rule
            #simple count of num of entity
            #or the sentence score
            #if count>0:
            temp_score, temp_num = sentence_score(score_table,temp_words,temp_labels)
            if temp_score>0.5 and temp_num>2:
                #write to the file
                for (word,label) in zip(temp_words,temp_labels):
                    f_output.write(word+u" "+label+'\n')
                f_output.write(u" "+'\n')
                #f_output.flush()
                
                #redefine the initial states
                frequency = frequency + 1
                count = 0
                temp_words = []
                temp_labels = []
    
    
    

    
    print (frequency)
    print(total_sentence)
    print (score_table)
    print (sentence_score(score_table,temp_words,temp_labels))
    print("The rate of selected sentence: ",float(frequency)/total_sentence)
    
    
file_path = 'phrase_translation/en-es-split-50d-phrase.txt'
output_path = 'better_translation_frequency/phrase_translation/en-es-split-50d-phrase.txt'

generate_frequency_table(file_path,output_path)

LofD = [{'name': 'johnny', 'surname': 'smith', 'age': 53},
 {'name': 'johnny', 'surname': 'ryan', 'age': 13},
 {'name': 'jakob', 'surname': 'smith', 'age': 27},
 {'name': 'aaron', 'surname': 'specter', 'age': 22},
 {'name': 'max', 'surname': 'headroom', 'age': 108},
]

y = {"red":3, "blue":4, "green":2, "yellow":5} 
frequencies = {key:float(value)/sum(y.values()) for (key,value) in y.items()}


    
    
#print(freqs(LofD))
