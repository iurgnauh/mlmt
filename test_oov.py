train_file = 'data/conll2003_train'
#embed_file = 'MUSE/en-es-origin-bigram-skipgram-512d/vectors-en.txt'
embed_file = 'embedding/enwiki.cbow.50d.es.txt'

train_set = set()
lines = open(train_file).readlines()
for line in lines:
	line = line.strip()
	if len(line) == 0:
		continue
	train_set.add(line.split()[0].lower())

print('Number tokens in train file:', len(train_set))

embed_word = set()
lines = open(embed_file).readlines()
for line in lines:
	if len(line.strip()) == 0:
		continue
	embed_word.add(line.strip().split()[0])

print('Number tokens in embedding file:', len(embed_word))

oov = set()
for w in train_set:
	if w not in embed_word:
		oov.add(w)

print(oov)
print('Total oov words {}, oov rates {}'.format(len(oov), len(oov) / len(train_set)))
