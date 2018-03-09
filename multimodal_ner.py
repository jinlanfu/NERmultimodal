from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Permute, Flatten, Dropout, TimeDistributedDense, Reshape, Layer, \
    ActivityRegularization, RepeatVector, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import History
from keras.layers import Input, Dense, Embedding, merge, Dropout, BatchNormalization
from keras.optimizers import SGD,Adagrad,Adam,RMSprop 
from keras.utils import np_utils
from keras.layers import ChainCRF
from keras import backend as K
import theano.tensor as T 
import cPickle
import h5py
import numpy as np

from load_data_multimodal import load_data
from ner_evaluate import evaluate,evaluate_each_class


def lambda_rev_gate(x):
	one = K.ones((sent_maxlen, final_w_emb_dim))
	rev_gate = one-x

	return rev_gate

def get_tag_index(pre_label,sent_maxlen,num_classes):
    sent_num = len(pre_label)
    pre_lab = pre_label.reshape(len(pre_label)*sent_maxlen,num_classes)
    pre_label_index = []
    for i in range(len(pre_lab)):
        list_pre = list(pre_lab[i])
        pre_label_index.append(list_pre.index(max(list_pre)))
    pre_label_index = np.reshape(pre_label_index, (sent_num,sent_maxlen))

    return pre_label_index

if __name__ == '__main__':
    """
    word_maxlen =30
    sent_maxlen = 35
    num_train = 4000
    num_dev = 1000
    num_test = 3257
    num_sent = 8257

    """

    print ('loading data...')
    id_to_vocb,word_matrix,sentences,datasplit,x, x_c, img_x, y, num_sentence, vocb, vocb_char, labelVoc = load_data()

    word_maxlen =30
    sent_maxlen = 35
    num_train = datasplit[1]
    num_dev = datasplit[2] - datasplit[1]
    num_test = datasplit[3] - datasplit[2]
    num_sent = len(sentences)
    y_ = y
    print 'num_train, num_dev, num_test: ', num_train, num_dev, num_test
    print 'num_sent', num_sent

    y = y.reshape((num_sent*sent_maxlen))
    x_c = x_c.reshape(len(x_c), sent_maxlen*word_maxlen)
    word_vocab_size = len(vocb) + 1
    char_vocab_size = len(vocb_char)+1
    num_classes = len(labelVoc)
    print "num_classes", num_classes

    y = np_utils.to_categorical(y, num_classes)
    y = y.reshape((num_sent, sent_maxlen, num_classes))

    # split the dataset into training set, validation set, and test set
    tr_x = x[:num_train]
    tr_x_c = x_c[:num_train]
    tr_y = y[:num_train]
    tr_img_x = img_x[:num_train]

    de_x = x[num_train:num_train+num_dev]
    de_x_c = x_c[num_train:num_train+num_dev]
    de_y = y[num_train:num_train+num_dev]
    de_img_x = img_x[num_train:num_train+num_dev]

    te_x = x[num_train+num_dev:]
    te_x_c = x_c[num_train+num_dev:]
    te_y = y[num_train+num_dev:]
    te_img_x =img_x[num_train+num_dev:]

    print('--------')
    print('Vocab size of word level:', word_vocab_size, 'unique words')
    print('Vocab size of char level:', char_vocab_size, 'unique characters')

    print('--------')
    print('x_[0], x_c[0], img_x[0].shape, y[0]')
    print(x[0], x_c[0], img_x[0].shape, y[0])

    print('--------')
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)
    print('x_cshape:', x_c.shape)

    w_emb_dim =200
    c_emb_dim = 30
    w_emb_dim_char_level = 50
    final_w_emb_dim = 200

    nb_epoch = 25
    batch_size = 10

    feat_dim = 512
    w = 7
    num_region = 49

    # build model
    print 'word_maxlen', word_maxlen
    print 'sent_maxlen', sent_maxlen 
    print "Build model..."

    # word level word representation
    w_tweet = Input(shape=(sent_maxlen,), dtype='int32')
    w_emb = Embedding(input_dim=word_vocab_size, output_dim=w_emb_dim,weights=[word_matrix], input_length=sent_maxlen, mask_zero=False)(
        w_tweet)
    w_feature = Bidirectional(LSTM(w_emb_dim, return_sequences=True, input_shape=(sent_maxlen, w_emb_dim)))(w_emb)

    # char level word representation
    c_tweet = Input(shape=(sent_maxlen*word_maxlen,), dtype='int32')
    c_emb = Embedding(input_dim=char_vocab_size, output_dim=c_emb_dim, input_length=sent_maxlen*word_maxlen, mask_zero=False)(
        c_tweet)
    c_reshape = Reshape((sent_maxlen, word_maxlen, c_emb_dim))(c_emb)
    c_conv1 = TimeDistributed(Convolution1D(nb_filter = 32, filter_length=2, border_mode='same', activation='relu'))(c_reshape)
    c_pool1 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv1)
    c_dropout1 = TimeDistributed(Dropout(0.25))(c_pool1)
    c_conv2 = TimeDistributed(Convolution1D(nb_filter =32, filter_length=3, border_mode ='same', activation = 'relu'))(c_dropout1)
    c_pool2 = TimeDistributed(MaxPooling1D(pool_length = 2))(c_conv2)
    c_dropout2 = TimeDistributed(Dropout(0.25))(c_pool2)
    c_conv3 = TimeDistributed(Convolution1D(nb_filter = 32, filter_length=4, border_mode='same', activation='relu'))(c_dropout2)
    c_pool3 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv3)
    c_dropout3 = TimeDistributed(Dropout(0.25))(c_pool3)
    c_batchNorm = BatchNormalization()(c_dropout3)
    c_flatten = TimeDistributed(Flatten())(c_batchNorm)
    c_fullConnect = TimeDistributed(Dense(100))(c_flatten)
    c_activate = TimeDistributed(Activation('relu'))(c_fullConnect)
    c_emb2 = TimeDistributed(Dropout(0.25))(c_activate)
    c_feature = TimeDistributed(Dense(w_emb_dim_char_level))(c_emb2)

    # merge the feature of word level and char level
    merge_w_c_emb = merge([w_feature,c_feature], mode = 'concat', concat_axis = 2)
    w_c_feature = Bidirectional(LSTM(output_dim=final_w_emb_dim, return_sequences = True))(merge_w_c_emb) 
    
    # reshape the image representation
    img = Input(shape=(1,feat_dim, w, w))
    img_reshape = Reshape((feat_dim, w * w))(img)
    img_permute = Permute((2, 1))(img_reshape)

    # word-guided visual attention 
    img_permute_reshape = TimeDistributed(RepeatVector(sent_maxlen))(img_permute) 
    img_permute_reshape = Permute((2, 1, 3))(img_permute_reshape) 
    w_repeat = TimeDistributed(RepeatVector(w*w))(w_c_feature) 
    w_repeat = TimeDistributed(TimeDistributed(Dense(final_w_emb_dim)))(w_repeat)
    img_permute_reshape = TimeDistributed(TimeDistributed(Dense(final_w_emb_dim)))(img_permute_reshape)
    img_w_merge = merge([img_permute_reshape, w_repeat], mode='concat') 

    att_w = TimeDistributed(Activation('tanh'))(img_w_merge)
    att_w = TimeDistributed(TimeDistributed(Dense(1)))(att_w) 
    att_w = TimeDistributed(Flatten())(att_w) 
    att_w_probability = Activation('softmax')(att_w) 

    img_permute_r = TimeDistributed(Dense(final_w_emb_dim))(img_permute)
    img_new = merge([att_w_probability, img_permute_r], mode='dot', dot_axes=(2,1)) 


    # image-guided textual attention
    img_new_dense = TimeDistributed(Dense(final_w_emb_dim))(img_new)  
    img_new_rep = TimeDistributed(RepeatVector(sent_maxlen))(img_new_dense) 

    tweet_dense = TimeDistributed(Dense(final_w_emb_dim))(w_c_feature) 
    tweet_dense1 = Flatten()(tweet_dense)
    tweet_rep = RepeatVector(sent_maxlen)(tweet_dense1) 
    tweet_rep = Reshape((sent_maxlen, sent_maxlen, final_w_emb_dim))(tweet_rep)

    att_img = merge([img_new_rep, tweet_rep], mode='concat') 
    att_img = TimeDistributed(Activation('tanh')) (att_img) 
    att_img = TimeDistributed(TimeDistributed(Dense(1)))(att_img) 
    att_img = TimeDistributed(Flatten())(att_img) 
    att_img_probability = Activation('softmax')(att_img)

    tweet_new = merge([att_img_probability, tweet_dense], mode='dot', dot_axes=(2, 1)) 

    img_new_resize = TimeDistributed(Dense(final_w_emb_dim, activation='tanh'))(img_new) 
    tweet_new_resize = TimeDistributed(Dense(final_w_emb_dim, activation='tanh'))(tweet_new) 


    # gate -> img new
    merge_img_w = merge([img_new_resize, tweet_new_resize], mode='sum')
    gate_img = TimeDistributed(Dense(1, activation='sigmoid'))(merge_img_w)
    gate_img = TimeDistributed(RepeatVector(final_w_emb_dim))(gate_img)  
    gate_img = TimeDistributed(Flatten())(gate_img) 
    part_new_img = merge([gate_img, img_new_resize], mode='mul') 


    #gate -> tweet new
    gate_tweet = Lambda(lambda_rev_gate, output_shape=(sent_maxlen, final_w_emb_dim))(gate_img)
    part_new_tweet = merge([gate_tweet, tweet_new_resize], mode='mul')
    
    part_img_w = merge([part_new_img, part_new_tweet], mode='concat')
    part_img_w = TimeDistributed(Dense(final_w_emb_dim))(part_img_w)


    #gate -> multimodal feature
    gate_merg = TimeDistributed(Dense(1, activation='sigmoid'))(part_img_w)
    gate_merg = TimeDistributed(RepeatVector(final_w_emb_dim))(gate_merg)  
    gate_merg = TimeDistributed(Flatten())(gate_merg) 
    part_sample = merge([gate_merg, part_img_w], mode='mul')

    w_c_emb = TimeDistributed(Dense(final_w_emb_dim))(w_c_feature) 

    merge_multimodal_w = merge([part_sample, w_c_emb], mode='concat') 
    multimodal_w_feature = TimeDistributed(Dense(num_classes))(merge_multimodal_w)


    crf = ChainCRF()
    crf_output = crf(multimodal_w_feature)
    model = Model(input=[w_tweet,c_tweet, img], output=[crf_output])

    rmsprop = RMSprop(lr=0.19, rho=0.9, epsilon=1e-08, decay=0.0)


    model.compile(loss=crf.loss, optimizer='rmsprop', metrics=['accuracy']) 


    label_test = y_[num_train+num_dev:]
    label_dev = y_[num_train:num_train+num_dev]


    print 'label_test shape',np.asarray(label_test).shape
    print 'label_dev shape',np.asarray(label_dev).shape

    max_f1 = 0
    for j in range(nb_epoch):
        model.fit([tr_x,tr_x_c, tr_img_x], tr_y,
        batch_size=batch_size,
        nb_epoch=1,verbose=1)

        pred_dev = model.predict([de_x,de_x_c, de_img_x], batch_size = batch_size, verbose=1,)
        pre_dev_label_index = get_tag_index(pred_dev, sent_maxlen, num_classes)
        acc_dev, f1_dev,p_dev,r_dev=evaluate(pre_dev_label_index, label_dev,de_x, labelVoc,sent_maxlen,id_to_vocb)
        print '##dev##, iter:',(j+1),'F1:',f1_dev,'precision:',p_dev,'recall:',r_dev

        if max_f1<f1_dev:
            max_f1 = f1_dev
            model.save_weights('../data/weights/multimodal_ner_best.h5')

    print 'the max dev F1 is:', max_f1
    model.load_weights('../data/weights/multimodal_ner_best.h5')
    pred_test = model.predict([te_x, te_x_c, te_img_x], batch_size = batch_size, verbose=1,)
    pre_test_label_index = get_tag_index(pred_test, sent_maxlen, num_classes)
    acc_test, f1_test,p_test,r_test=evaluate(pre_test_label_index, label_test,te_x,labelVoc,sent_maxlen,id_to_vocb)
    pre_test_label_index_2 = pre_test_label_index.reshape(len(label_test)*sent_maxlen)
    print '----------'
    print '##test##, evaluate:''F1:',f1_test,'precision:',p_test,'recall:',r_test

    #evaluate each class
    for class_type in ('PER', 'LOC', 'ORG', 'OTHER'):
        f1_t_cl,p_t_cl,r_t_cl =evaluate_each_class(pre_test_label_index, label_test,te_x,labelVoc,sent_maxlen,id_to_vocb, class_type)
        print 'class type:', class_type, 'F1:',f1_t_cl,'precision:',p_t_cl,'recall:',r_t_cl
