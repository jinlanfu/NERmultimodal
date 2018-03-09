import codecs
import os
from collections import Counter
import cPickle
import h5py
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from numpy import random
from gensim.models import word2vec
from word2vecReader import Word2Vec
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

data_dir = '../data'
tweet_data_dir = '../data/tweet_new/'
# h5_file = "img_vgg_feature_all.h5"
h5_file = 'img_vgg_feature_224.h5'
IMAGEID='IMGID'
img_feature_file = h5py.File(os.path.join(data_dir, h5_file), 'r')

def load_sentence(IMAGEID, tweet_data_dir, train_name, dev_name, test_name):
	"""
	read the word from doc, and build sentence. every line contain a word and it's tag
	every sentence is split with a empty line. every sentence begain with an "IMGID:num"

	"""
	#IMAGEID='IMGID'
	img_id = []
	sentences = []
	sentence = []
	sent_maxlen =0
	word_maxlen =0
	img_to_feature = []
	datasplit = []

	for fname in (train_name, dev_name, test_name):
		datasplit.append(len(img_id))
		with open(os.path.join(tweet_data_dir, fname), 'r') as file:
			for line in file:
				line = line.rstrip()
				if line == '':
					sent_maxlen = max(sent_maxlen, len(sentence))
					sentences.append(sentence)
					sentence = []
				else:
					if IMAGEID in line:
						num =  line[6:]
						img_id.append(num)
					else:
						sentence.append(line.split('\t'))
						word_maxlen = max(word_maxlen, len(str(line.split()[0])) )

        sentences.append(sentence)
        datasplit.append(len(img_id))
	num_sentence = len(sentences)

	print "datasplit",datasplit
	print sentences[len(sentences)-2]
	print sentences[0]
#    train_num = datasplit[1]
#    dev_num = datasplit[2]-datasplit[1]
#    test_num = datasplit[3]-datasplit[2]

	## get image feature.
	for image in img_id:
		feature = img_feature_file.get(image)
		np_feature = np.array(feature)
		img_to_feature.append(np_feature)

	print 'sent_maxlen', sent_maxlen
	print 'word_maxlen', word_maxlen
	print 'number sentence', len(sentences)
	print 'number image', len(img_id)
	return [datasplit,sentences, img_to_feature, sent_maxlen,word_maxlen, num_sentence]

def load_word_matrix(vocabulary, size=200):
	"""
		This function is used to convert words into word vectors
	"""
	b = 0
	word_matrix = np.zeros((len(vocabulary)+1, size))
	model = word2vec.Word2Vec.load('../40Wtweet_200dim.model')
	for word, i in vocabulary.iteritems():
		try:
			word_matrix[i]=model[word.lower().encode('utf8')]
		except KeyError:
			# if a word is not include in the vocabulary, it's word embedding will be set by random.
			word_matrix[i] = np.random.uniform(-0.25,0.25,size)
			b+=1	
	print('there are %d words not in model'%b)
	return word_matrix

def vocab_bulid(sentences):
	"""
	input: 
		sentences list, 
		the element of the list is (word, label) pair.
	output: 
		some dictionaries.

	"""
	words = []
	chars = []
	labels = []

	for sentence in sentences:
		for word_label in sentence:
			words.append(word_label[0])
			labels.append(word_label[1])
			for char in word_label[0]:
				chars.append(char)
	word_counts = Counter(words)
	vocb_inv = [x[0] for x in word_counts.most_common()]
	vocb = {x: i + 1 for i, x in enumerate(vocb_inv)}
	vocb['PAD'] =0
	id_to_vocb = {i:x for x,i in vocb.items()}

	char_counts = Counter(chars)
	vocb_inv_char = [x[0] for x in char_counts.most_common()]
	vocb_char = {x: i + 1 for i, x in enumerate(vocb_inv_char)}

	labels_counts = Counter(labels)
	print 'labels_counts',len(labels_counts)
	print labels_counts
	labelVoc_inv, labelVoc = label_index(labels_counts)
	print 'labelVoc',labelVoc
	
	return [id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv]

def label_index(labels_counts):
	"""
	   the input is the output of Counter. This function defines the (label, index) pair, 
	   and it cast our datasets label to the definition (label, index) pair. 
	"""

	num_labels = len(labels_counts)
	labelVoc_inv = [x[0] for x in labels_counts.most_common()]

	labelVoc = {'0':0,
	'B-PER':1, 'I-PER':2,
	'B-LOC':3, 'I-LOC':4,
	'B-ORG':5, 'I-ORG':6,
	'B-OTHER':7, 'I-OTHER':8,
	'O':9}
	if len(labelVoc) < num_labels:
		for key,value in labels_counts.items():
			if not labelVoc.has_key(key):
				labelVoc.setdefault(key, len(labelVoc))
	return labelVoc_inv, labelVoc


def pad_sequence(sentences, img_to_feature, vocabulary, vocabulary_char, labelVoc, word_maxlen=30, sent_maxlen=35):
	"""
		This function is used to pad the word into the same length, the word length is set to 30.
		Moreover, it also pad each sentence into the same length, the length is set to 35.

	"""

	print sentences[0]
	x = []
	y = []
	for sentence in sentences:
		w_id =[]
		y_id = []
		for word_label in sentence:
			w_id.append(vocabulary[word_label[0]])
			y_id.append(labelVoc[word_label[1]])
		x.append(w_id)
		y.append(y_id)

	y = pad_sequences(y, maxlen=sent_maxlen).astype(np.int32)
	x = pad_sequences(x, maxlen=sent_maxlen).astype(np.int32)
	
	img_x = np.asarray(img_to_feature)
	
	x_c = []
	for sentence in sentences:
		s_pad = np.zeros([sent_maxlen, word_maxlen], dtype = np.int32)
		s_c_pad = []
		for word_label in sentence:
			w_c = []
			char_pad = np.zeros([word_maxlen], dtype=np.int32)
			for char in word_label[0]:
				w_c.append(vocabulary_char[char])	
			if len(w_c) <= word_maxlen:
				char_pad[:len(w_c)] =w_c
			else:
				char_pad = w_c[:word_maxlen]

			s_c_pad.append(char_pad)
			
		for i in range(len(s_c_pad)):
			s_pad[sent_maxlen-len(s_c_pad)+i,:len(s_c_pad[i])] =s_c_pad[i]
		x_c.append(s_pad)

	x_c = np.asarray(x_c)
	x = np.asarray(x)
	y = np.asarray(y)

	return [x, x_c, img_x, y]



def load_data():
	# vocabFileName = 'vocabulary_full_tweet_coatt_img.pkl'
	# if os.path.isfile(vocabFileName):
	# 	print "loading vocabulary..."
	# 	id_to_vocb,word_matrix,sentences,datasplit,x, x_c, img_x, y, sent_maxlen, word_maxlen, num_sentence, vocb, vocb_char, labelVoc = cPickle.load(open(vocabFileName))
	# else:
	print 'calculating vocabulary...'
	datasplit,sentences, img_to_feature, sent_maxlen,word_maxlen, num_sentence = load_sentence(
		IMAGEID, tweet_data_dir, 'train', 'dev', 'test')
	id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv =vocab_bulid(sentences)
	word_matrix = load_word_matrix(vocb, size=200)
	x, x_c, img_x, y = pad_sequence(sentences, img_to_feature, vocb, vocb_char, labelVoc, word_maxlen=30, sent_maxlen=35)
	# cPickle.dump([id_to_vocb,word_matrix,sentences,datasplit,x, x_c, img_x, y, sent_maxlen, word_maxlen, num_sentence, vocb, vocb_char, labelVoc], open(vocabFileName, "w"))
	return 	[id_to_vocb,word_matrix,sentences,datasplit,x, x_c, img_x, y, num_sentence, vocb, vocb_char, labelVoc]

if __name__ == '__main__':
	load_data()
	id_to_vocb,word_matrix,sentences,datasplit,x, x_c, img_x, y, sent_maxlen, word_maxlen, num_sentence, vocb, vocb_char, labelVoc = load_data()
	print x_c[0][0][1]
	print x_c[0][0]
	print x_c[1]
	print img_x[0]
	print np.asarray(img_x).shape
	print y
	print 'datasplit', datasplit
