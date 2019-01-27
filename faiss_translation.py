import faiss                   # make faiss available
import numpy as np
import time
import sys
import io

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id



#we need to have query and database
d = 50                           # dimension for word embedding
#nb = 100000                      # database size
#nq = 100000                       # nb of queries


src_path = 'embedding/enwiki.cbow.50d.es.txt'
tgt_path = 'embedding/eswiki.cbow.50d.en.txt'

nmax = 100000  # maximum number of word embeddings to load

src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax) #query (English)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax) #database (chinese)

#must be numpy array
#convert to numpy
src_embeddings = np.array(src_embeddings,dtype ='float32' )
tgt_embeddings = np.array(tgt_embeddings,dtype ='float32' )


start_time = time.time()

index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(tgt_embeddings)                  # add database vectors to the index
print(index.ntotal)
print("--- %s seconds ---" % (time.time() - start_time))



start_time = time.time()

k = 1     #get the top 1 neighbour
D, Translation = index.search(src_embeddings, k)     # actual search
print(Translation[:5])                   # neighbors of the 5 first queries
print(Translation[-5:])                  # neighbors of the 5 last queries
print("--- %s seconds ---" % (time.time() - start_time))


print(type(Translation[:5]))
#print the translation index
'''
for elements in Translation[:5]:
    for index in elements:
        print (tgt_id2word[index])
'''
print("--- %s seconds ---" % (time.time() - start_time))


#do the actual transation from English file to chinese file




def translate(file_path,Translation,src_id2word,tgt_id2word):
    #buffer size?
    # f_output = io.open('ontonotes_translation/en-zh-cheap_train-MUSE.txt', 'w', encoding='utf-8')

    f_output = io.open('new_translation/en-es-cheap_train-test.txt', 'w', encoding='utf-8')
    in_lines = io.open(file_path,'r',encoding='utf8').readlines()
    index = 0
    oov_count = 0
    oov_hyphen = 0
    oov_set = []
    
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
                    #change the seperator for chinese
                    f_output.write(tgt_id2word[idx]+" "+label+'\n')
                    #print(ele)
            else:
                #otherwise, just copy the word from source language
                #change the seperator for chinese
                f_output.write(word+" "+label+'\n')
                oov_count += 1
                if '-' in word:
                    oov_hyphen += 1
                else:
                    oov_set.append(word)
                #f_output.write(word+u" "+label+'\n')
                #print ("out of vocab word: ",word)
            index = index + 1
        elif len(line) < 2:
            #change the seperator for chinese
            f_output.write(" "+'\n')
            #f_output.write(u" "+'\n')
            #You can also force flush the buffer to a file programmatically with the flush() method.
            f_output.flush()
        if index % 1000==0:
            print (index)
        #to do add sentence spliter here:
    f_output.close()
    print('Total OOV number:', oov_count)
    print('OOV caused by hyphen:', oov_hyphen)
    print('\n'.join(oov_set))


    
#bug here?
#this is the source file path?
#always english file
file_path = "data/conll2003_train"
translate(file_path,Translation,src_id2word,tgt_id2word)



'''
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

start_time = time.time()

index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
print("--- %s seconds ---" % (time.time() - start_time))

k=4                                   # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)



start_time = time.time()

D, Translation = index.search(xq, k)     # actual search
print(Translation[:5])                   # neighbors of the 5 first queries
print(Translation[-5:])                  # neighbors of the 5 last queries
print("--- %s seconds ---" % (time.time() - start_time))

#I index can be understood as translation table
#if word in range, then otherwise just copy
#
'''

