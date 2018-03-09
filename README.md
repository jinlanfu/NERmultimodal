# NERmultimodal
## 1.Introduction 
Keras Implementation of "Adaptive Co-attention Network for Named Entity Recognition in Tweets". The multimodal dataset used in our paper can be download from the repository. A more perfected version of the datasets will be released later. 

## 2. Requirements
1) Python 2.7 or higher
2) Keras 1.2 (the vesion including the CRF model), the backend is theano.
3) The image features were extracted from 16-layer VGGNet. Before extracting the features, you need to download a pretrained model -- vgg16_weights_th_dim_ordering_th_kernels_notop.h5.
4) Moreover, you need to download the word embedding trained by tweets from http://pan.baidu.com/s/1boSlljL. 

## 3. Data Format
Our datasets include 8,257 tweet and image pairs. We split the dataset into three parts: the training set, development set, and testing set, which contain 4,000, 1,000, and 3,257 tweets, respectively.  

We set an image id for each picture, and we put this image id in the begging of a tweet. We use a blank line to split a sample. Here's an example of such a file:

IMGID:50447
RT	O
@washingtonpost	O
:	O
Two	O
maps	O
that	O
show	O
the	O
shocking	O
inequality	O
in	O
Baltimore	B-LOC
http://t.co/FssPdKxglv	O
http://t.co/JZiqXSTNec	O

IMGID:418340
Rep	O
.	O
Howard	B-PER
Coble	I-PER

## 4. Usage
1) Extracting the image features:
	$ python vgg_image_feature.py
2) Training the model:
	$ python multimodal_ner.py

## 5. Evalution
The Evaluation code to calculate F1, Precision, and recall is in ner_evaluate.py. 

## 6. Citation
If you find the implementation or datasets useful, please cite the following paper: 
@article{zhang2018adaptive,
  title={Adaptive Co-attention Network for Named Entity Recognition in Tweets},
  author={Zhang, Qi and Fu, Jinlan and Liu, Xiaoyu and Huang, Xuanjing},
  year={2018}
  publisher={AAAI}
}.
