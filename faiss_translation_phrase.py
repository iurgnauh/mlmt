import faiss  # make faiss available
import numpy as np
import time
import sys
import io


def load_vec(emb_path, nmax=50000, load_all_phrase=False):
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
            if len(word2id) >= nmax:
                if load_all_phrase:
                    if '<_>' in word:
                        # assert word not in word2id, '{} word found twice'.format(word)
                        if word in word2id:  continue
                        vectors.append(vect)
                        word2id[word] = len(word2id)
                    continue
                else:
                    break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


# we need to have query and database
d = 50  # dimension for word embedding
# nb = 100000                      # database size
# nq = 100000                       # nb of queries


# src_path = 'MUSE/en-es/vectors-en.txt'
# tgt_path = 'MUSE/en-es/vectors-es.txt'
src_path = 'MUSE/en-es-50d-phrase/vectors-en.txt'
tgt_path = 'MUSE/en-es-50d-phrase/vectors-es.txt'

nmax = 100000  # maximum number of word embeddings to load

src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)  # query (English)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)  # database (chinese)
print("Total words in src_lang: ", len(src_id2word))
print("Total words in tgt_lang: ", len(tgt_id2word))

# must be numpy array
# convert to numpy
src_embeddings = np.array(src_embeddings, dtype='float32')
tgt_embeddings = np.array(tgt_embeddings, dtype='float32')

start_time = time.time()

index = faiss.IndexFlatL2(d)  # build the index
print(index.is_trained)
index.add(tgt_embeddings)  # add database vectors to the index
print(index.ntotal)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

k = 1  # get the top 1 neighbour
D, Translation = index.search(src_embeddings, k)  # actual search
print(Translation[:5])  # neighbors of the 5 first queries
# for i in range(10000):
#     if '<_>' in src_id2word[i]:
#         print(src_id2word[i], tgt_id2word[Translation[i][0]])
#print(src_id2word[:5])
#print(tgt_id2word[Translation[:5]])
print(Translation[-5:])  # neighbors of the 5 last queries
print("--- %s seconds ---" % (time.time() - start_time))

print(type(Translation[:5]))
# print the translation index
'''
for elements in Translation[:5]:
    for index in elements:
        print (tgt_id2word[index])
'''
print("--- %s seconds ---" % (time.time() - start_time))


def get_sentences(lines):
    sentences = []
    sent = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if len(sent) != 0:
                sentences.append(sent)
            sent = []
        else:
            sent.append(line)
    return sentences


def translate(file_path, Translation, src_id2word, tgt_id2word, ngram=2):
    # buffer size?
    f_output = io.open('phrase_translation/en-es-50d-phrase.txt', 'w', encoding='utf-8')
    f_output_split = io.open('phrase_translation/en-es-split-50d-phrase.txt', 'w', encoding='utf-8')

    in_lines = io.open(file_path, 'r', encoding='utf8').readlines()

    word2id = {v: k for k, v in src_id2word.items()}
    sentences = get_sentences(in_lines)
    print("Total number of sentences:", len(sentences))
    phrase_count = 0
    oov_count = 0
    oov_hyphen = 0
    oov_set = []
    num_split = 0
    num_combine = 0
    num_phrase = 0
    for sent in sentences:
        new_sent = []
        for line in sent:
            word = line.strip().split()[0]
            tag = line.strip().split()[-1]            
            if '-' not in word:
                new_sent.append({'word': word, 'tag': tag})
                continue
            words = word.split('-')
            if '<_>'.join(words) in word2id:
                num_phrase += 1
                new_sent.append({'word': '<_>'.join(words), 'tag': tag})
                #continue
            else:
                new_sent.append({'word': word, 'tag': tag})
            """
            if ''.join(words) in word2id:
                num_combine += 1
                new_sent.append({'word': ''.join(words), 'tag': tag})
            else:
                not_found = True
                for w in words:
                    if w in word2id:
                        not_found = False
                        break
                if not_found:
                    new_sent.append({'word': word, 'tag': tag})
                    continue
                written = 0
                for w in words:
                    tag_continue = tag
                    if tag_continue[0] == 'B':
                        tag_continue = tag_continue.replace('B', 'I', 1)
                    if len(w) != 0:
                        new_sent.append({'word': w, 'tag': tag if written == 0 else tag_continue})
                        written += 1
                num_split += 1
            new_sent.append({'word': word, 'tag': tag})
            """
        sent = new_sent
        i = 0
        while i < len(sent):
            cur_ngram = min(len(sent) - i, ngram)
            while cur_ngram >= 1:
                tags = [token['tag'] for token in sent[i:i+cur_ngram]]
                if not (tags[0] == tags[-1] or tags[0].split('-')[-1] == tags[-1].split('-')[-1]):
                    cur_ngram -= 1
                    continue
                words = [token['word'] for token in sent[i:i+cur_ngram]]
                if '<_>'.join(words) in word2id or '<_>'.join(words).lower() in word2id:
                    if '<_>'.join(words) in word2id:
                        word_index = word2id['<_>'.join(words)]
                    else:
                        word_index = word2id['<_>'.join(words).lower()]
                    neibor_index = Translation[word_index][0]
                    tgt_words = tgt_id2word[neibor_index]
                    for j, tgt_word in enumerate(tgt_words.split('<_>')):
                        tag_continue = tags[-1]
                        if tag_continue[0] == 'B':
                            tag_continue = tag_continue.replace('B', 'I', 1)
                        f_output_split.write(tgt_word + ' ' + (tags[j] if j < cur_ngram else tag_continue) + '\n')
                    f_output.write(tgt_words + ' ' + tags[0] + '\n')
                    if len(words) > 1 or '<_>' in words[0]:
                        print('EN:', words, 'ES:', tgt_words)
                        phrase_count += 1
                    break
                if cur_ngram > 1:
                    cur_ngram -= 1
                    continue
                oov_count += 1
                if '-' in words[0]:
                    oov_hyphen += 1
                """
                else:
                    punc = [',', '.', '(', ')', '"', "'", '[', ']', ':']
                    if words[0] not in punc and (words[0][0] < '0' or words[0][0] > '9'):
                        # oov_count += 1
                        oov_set.append(words[0])
                """
                f_output.write(words[0] + ' ' + tags[0] + '\n')
                f_output_split.write(words[0] + ' ' + tags[0] + '\n')
                break
            i += cur_ngram
        f_output_split.write('\n')
        f_output.write('\n')

    f_output.close()
    f_output_split.close()

    print('Total phrase translated:', phrase_count)
    print('Total OOV count:', oov_count)
    print('OOV caused by hyphen:', oov_hyphen)
    # print('\n'.join(oov_set))
    print('Number phrase:', num_phrase)
    print('Number combine:', num_combine)
    print('Number split:', num_split)


# bug here?
# this is the source file path?
# always english file
file_path = "data/conll2003_train"
translate(file_path, Translation, src_id2word, tgt_id2word)

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

