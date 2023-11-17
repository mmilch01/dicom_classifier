#!/usr/bin/env python
# coding: utf-8
import csv, getpass, io, json, os, pickle, pydicom, re, shlex, subprocess, sys, tempfile, tensorflow as tf, unittest, warnings, zipfile, argparse, numpy as np, math,sklearn

# Subpackages and function imports
#from IPython.display import FileLink
#from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, GRU, Input, LSTM, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from xnat_utils import *
from autoencoder_classifier import DICOMAutoencoderModel

"""
@author: mmilchenko
"""
class AttentionModel: 
    def __init__(self,daem:DICOMAutoencoderModel=None):
        if daem is None: daem=DICOMAutoencoderModel()
        self.daem=daem

    def load_autoencoder_from_dir(self,tdir):
        #self.sequences=daem.read_pkl(tdir+'sequences.pkl')
        self.tokenizer=self.daem.read_pkl(tdir+'tokenizer.pkl')
        self.sequences=self.daem.read_pkl(tdir+'sequences.pkl')


    def save_autoencoder_to_dir(self,tdir):
        self.daem.write_pkl(self.sequences, tdir+'sequences.pkl')
        self.daem.write_pkl(tokenizer, tdir+'tokenizer.pkl')

    def sequences_from_scans(self,scans,max_length,fit_tokenizer=False):
        #scans=daem.read_pkl_scans('/data/ImagingCommons/data.ImagingCommons.416237_1.pkl')
        #scans=daem.read_pkl_scans('./test/all_scans_hofid.pkl')
        #scans=daem.read_pkl_scans('/data/mirrir_1351062.pkl')
        #scans=self.daem.read_pkl_scans(scans_pkl_file)
        descs=self.daem.filter_least_frequent_words(self.daem.prepare_descs(scans),15000)
        if fit_tokenizer:
            self.tokenizer = Tokenizer(filters='',oov_token='UNK')
            self.tokenizer.fit_on_texts(descs)
            
        self.sequences = self.tokenizer.texts_to_sequences(descs)
        vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(seq) for seq in self.sequences)        
        return pad_sequences(self.sequences, maxlen=max_length, padding='post')

    def create_autoencoder_model(self):
        encoder_input = Input(shape=(max_length,))
        embedding_layer = Embedding(vocab_size, 32, input_length=max_length)(encoder_input)
        attention = MultiHeadAttention(num_heads=2, key_dim=32)(embedding_layer, embedding_layer)
        encoder_output = Dense(50, activation="relu")(attention)

        decoder_input = Input(shape=(max_length,))
        decoder_embedding = Embedding(vocab_size, 32)(decoder_input)
        decoder_attention = MultiHeadAttention(num_heads=2, key_dim=32)(decoder_embedding, encoder_output)
        decoder_output = Dense(vocab_size, activation='softmax')(decoder_attention)

        self.autoencoder = Model([encoder_input, decoder_input], decoder_output)
        self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.autoencoder

    def data_generator(self,X,batch_size=32):
        while True:
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_X = X[start:end]
                batch_Y = to_categorical(batch_X, num_classes=vocab_size)
                yield [batch_X, batch_X], batch_Y
        
    def train_model(self,max_length):                    
        X=pad_sequences(self.sequences, maxlen=max_length, padding='post')
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        self.autoencoder.summary()
        batch_size = 128
        steps_per_epoch = np.ceil(len(X_train) / batch_size)
        reducer = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=10)
        autoencoder.fit_generator(self.data_generator(X_train, batch_size), epochs=1000, steps_per_epoch=steps_per_epoch,callbacks=[reducer])

    def save_autoencoder(self,tdir,model_file='autoencoder.dualhead.21ep.h5'):
        self.autoencoder.save(tdir+'autoencoder.dualhead.21ep.h5')
        
    def load_autoencoder(self,tdir,model_file='autoencoder.dualhead.21ep.h5'):
        self.autoencoder=tf.keras.models.load_model(tdir+'autoencoder.dualhead.21ep.h5')
        return self.autoencoder

    def evaluate_autoencoder(self,model, X_test, batch_size=32):
        steps = np.ceil(len(X_test) / batch_size)
        test_generator = self.data_generator(X_test, batch_size)
        loss, accuracy = model.evaluate(test_generator, steps=steps)
        return loss, accuracy

    def evaluate_autoencoder_test(self,model,X):
        # Call the function with the model and X_test
        test_loss, test_accuracy = evaluate_model(model, X)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

    def sentence_to_tokens(self,sentence, tokenizer, max_length):
        """
        Convert a sentence to a sequence of tokens using a tokenizer.
        """
        # Tokenize the sentence
        sequence = self.tokenizer.texts_to_sequences([sentence])[0]
        # Pad the sequence
        padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        return padded_sequence

    def tokens_to_sentence(self,tokens, tokenizer):
        """
        Convert a sequence of tokens to a sentence using a tokenizer's index-to-word mapping.
        """
        # Convert tokens to words using the tokenizer's index_word dictionary
        words = [tokenizer.index_word[token] for token in tokens if token != 0]
        # Join words into a sentence
        sentence = ' '.join(words)
        return sentence

    def predict_sentence(self, sentence, model, tokenizer, max_length):
        """
        Get the model's output for a given input sentence and convert it back to a sentence.
        """
        # Convert input sentence to tokens
        input_tokens = self.sentence_to_tokens(sentence, tokenizer, max_length)
        # Get model's output probability distribution for next tokens
        output_probs = model.predict([input_tokens, input_tokens])[0]
        # Convert probability distribution to tokens (argmax over the vocabulary for each time step)
        output_tokens = np.argmax(output_probs, axis=-1)
        # Convert output tokens to sentence
        output_sentence = tokens_to_sentence(output_tokens, tokenizer)
        return output_sentence

    def predict_sentence_test(self,descs):
        input_sentence = descs[0]
        output_sentence = predict_sentence(input_sentence, self.autoencoder, self.tokenizer, self.max_length)
        print(f"Input Sentence: {input_sentence}")
        print(f"Output Sentence: {output_sentence}")

    def load_training_scans(self,file):
        scans1=self.daem.read_pkl_scans(file)
        descs1=self.daem.prepare_descs(scans1)
        return scans1,descs1
        
    def load_training_scans_test(self):
        scans1,descs1=self.load_training_scans('./test/all_scans_hofid.pkl')
        input_sentence = descs1[10000]
        output_sentence = predict_sentence(input_sentence, autoencoder, self.tokenizer, max_length)
        print(f"Input Sentence: {input_sentence}")
        print(f"Output Sentence: {output_sentence}")
        
    def csv_to_dict(self,filename):
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            data_dict = {field: [] for field in reader.fieldnames}
            for row in reader:
                for field in reader.fieldnames:
                    data_dict[field].append(row[field])
        return data_dict

    def load_scans_csv_test(self,filename):
        filename = '/home/mmilchenko/src/scan_classifier/test/all_scans_hofid.csv'
        scans_xnat = self.csv_to_dict(filename)
        td='/home/mmilchenko/src/scan_classifier/test/dcm'
        scan_types=reorder_labels(td,scans_xnat['hof_id'])
        #Y_label=scan_types
        return scan_types
        
    def reorder_labels(self,directory_name, label_array):
        '''
        reorder labels read from csv file, to match actual DICOM files in target directory
        '''
        # Get list of all files in directory
        files = [f for f in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, f))]
        
        # Extract the six-digit numbers from file names using regex
        file_numbers = [int(re.match(r'(\d{6})_', f).group(1)) for f in files if re.match(r'(\d{6})_', f)]
        
        # Sort files based on their numbers
        sorted_file_numbers = sorted(file_numbers)
        
        # Create a new label array using the sorted file numbers
        new_label_array = [label_array[num] for num in sorted_file_numbers]
        
        return new_label_array

    def create_categorical_labels_test(self,Y_label):
        labels=["CBF", "CBV", "DSC", "DWI", "FA", "MD", "MPRAGE", "MTT", "OT", "PBP", "SWI", "T1hi", "T1lo", "T2FLAIR", "T2hi", "T2lo", "TRACEW", "TTP"]
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(Y_label)
        categorical_labels = to_categorical(encoded_labels)
        return categorical_labels


    def load_json(self, json_file):
        with open(json_file, 'r') as fp:
            out_dict=json.loads(fp.read())
        return out_dict
    def save_json(self, var, file):
        with open(file,'w') as fp:
            json.dump(var, fp)
    
    def load_nomenclature(self,json_file):
        return self.load_json(json_file)['scan_types']        

    def get_classifier_training_sequences_test(self,descs):
        categorical_labels=self.create_categorical_labels_test()
        sequences1 = self.tokenizer.texts_to_sequences(descs)
        X1 = pad_sequences(sequences1, maxlen=max_length, padding='post')        
        X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, categorical_labels, test_size=0.2, random_state=42)
        return X1_train,X1_test,Y1_train,Y1_test

    def create_classifier_model(self):
        embedding_layer = self.autoencoder.layers[1]
        embedding_layer.trainable = False

        input_layer = Input(shape=(self.max_length,))
        embedding_output = embedding_layer(input_layer)

        attention_output = MultiHeadAttention(num_heads=2,key_dim=50)(embedding_output,embedding_output)
        encoder_output=Dense(50, activation="relu")(attention_output)
        flattened_output=Flatten()(encoder_output)
        dense1=Dense(100,activation='relu')(flattened_output)
        x=Dropout(0.5)(dense1)
        output_layer = Dense(categorical_labels.shape[1], activation='softmax')(x)

        self.classification_model = Model(inputs=input_layer, outputs=output_layer)
        self.classification_model.summary()        
        self.classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.classification_model

    def train_classifier_model(self,X1_train,Y1_train):
        self.classification_model.optimizer.learning_rate.assign(0.001)
        reducer = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=10)
        self.classification_model.fit(X1_train, Y1_train, epochs=100, batch_size=512, validation_split=0.2,callbacks=[reducer])

    def evaluate_classifier_model(self,X1_test,Y1_test):
        loss, accuracy = self.classification_model.evaluate(X1_test, Y1_test)

    def load_model(self,tokenizer_file,model_file):
        try:
            print('loading tokenizer from',tokenizer_file)
            self.tokenizer=self.daem.read_pkl(tokenizer_file)
            print('loading mdoel from',model_file)
            model=tf.keras.models.load_model(model_file)
        except Exception as e:
            print('error loading: '+zipfile)
            print(e)
        return self.tokenizer,model
            
    def save_model(self, tokenizer, model, root):
        try: 
            tokenizer_file,h5file=root+'.tokenizer',root+'.h5'    
            #save individual files for vectorizer and model
            print('saving tokenizer to',tokenizer_file)
            self.daem.write_pkl(tokenizer,tokenizer_file)        
            print('saving model to',h5file)
            model.save(h5file)
            
        except Exception as e:
            print('error saving:',zipfile)
            print(e)
    
    def print_misclassified_cases(self,X1_test):
        pred=self.classification_model.predict(X1_test)
        pred_ord=np.argmax(pred,1)
        #print(pred_ord)
        pred_inv=self.label_encoder.inverse_transform(pred_ord)
        val_ord=self.label_encoder.inverse_transform(np.argmax(Y1_test,1))
        m,n=0,0
        for i in range (0,len(pred_inv)):
            m+=1
            if val_ord[i] != pred_inv[i]: 
                s=tokens_to_sentence(X1_test[i],self.tokenizer).split(' ')        
                #print(i,val_ord[i],pred_inv[i])
                print(i,val_ord[i],pred_inv[i],' '.join([ s[i] for i in range(0,len(s)) if s[i].startswith('seriesdescription') ]))
                #print(tokens_to_sentence(X1_test[i],self.tokenizer))
                n+=1        
        print('misclassifications: {} out of {}'.format(n,m))

    def classify_dicom_scans(self, dcm_file_list, tokenizer_file, model_file, nomenclature_file,confidence_level=0.95,unknown_label='UNKNOWN'):
        scans=self.daem.scans_from_files(dcm_file_list)
        self.tokenizer,self.classification_model=self.load_model(tokenizer_file,model_file)
        max_length=self.classification_model.layers[0].input_shape[0][1]
        sd=self.sequences=self.sequences_from_scans(scans,max_length)
        #print(self.sequences.shape)
        #print(self.sequences)
        
        scan_types=self.load_nomenclature(nomenclature_file)
        pred=self.classification_model.predict(self.sequences)
        
        pred_ord=np.argmax(pred,1)
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(self.load_nomenclature(nomenclature_file))                                    
        pred_inv=label_encoder.inverse_transform(pred_ord)
        out_label=[]
        series_descriptions=[]
        #prediction quality metrics
        pred_entropy=[]
        pred_gini_impurity=[]
        pred_margin_confidence=[]
        
        for i in range (0,len(pred_inv)):
            pred_cur=pred[i]
            max_pred=np.argmax(pred_cur)
            if pred_cur[max_pred]>=confidence_level: 
                out_label+=[pred_inv[i]]
            else:
                out_label+=[unknown_label]
                            
            pred_gini_impurity+=[1-np.sum(np.array([pred_cur[i]*pred_cur[i] for i in range (0,len(pred_cur))]))]
            pred_margin_confidence+=[np.partition(pred_cur, -2)[-1]-np.partition(pred_cur, -2)[-2]]
            series_descriptions+=[scans[i]['SeriesDescription'].replace(' ','_')+' ']
        print(out_label)        
        return out_label,pred_gini_impurity,pred_margin_confidence,series_descriptions
        
def parse_args():
    parser = argparse.ArgumentParser(description='Classify a list of DICOM files using a trained model.')
    parser.add_argument('dicom_files', type=str, nargs='+', help='List of paths to DICOM files to be classified.')
    parser.add_argument('--model_file', type=str, help='trained model file')
    parser.add_argument('--tokenizer_file', type=str, help='tokenizer file')
    parser.add_argument('--nomenclature_file', type=str, help='nomenclature file')    
    return parser.parse_args()

def l2str(arr):
    lst=[str(arr[i]) for i in range(0,len(arr))]
    return ' '.join(lst)

def main():
    args = parse_args()
    dicom_files = args.dicom_files
    model_file = args.model_file
    tokenizer_file = args.tokenizer_file
    nomenclature_file = args.nomenclature_file
    
    # Verify that the specified DICOM files exist
    for file in dicom_files:
        if not os.path.exists(file):
            print(f"Error: Specified DICOM file does not exist: {file}", file=sys.stderr)
            sys.exit(1)

    for file in (model_file,tokenizer_file,nomenclature_file):
        if not os.path.exists(file):
            print(f"Error: Specified file does not exist: {file}", file=sys.stderr)
            sys.exit(1)
        
    am=AttentionModel()
    #print('classify_dicom_scans {} {} {}'.format(dicom_files,tokenizer_file,model_file,nomenclature_file))
    labels,pred_gini_impurity,pred_margin_confidence,series_descriptions=am.classify_dicom_scans(dicom_files, tokenizer_file, model_file, nomenclature_file,confidence_level=0.5)
    
#    with open('classification_output.txt','wt') as f:
#        print('files=({})'.format(' '.join(dicom_files)),file=f)
#        print('labels=({})'.format(' '.join(labels)),file=f)
#        print('series_descriptions=({})'.format(' '.join(series_descriptions)),file=f)
        #print('pred_entropy=({})'.format(' '.join(pred_entropy)),file=f)
#        print('pred_gini_impurity=({})'.format(l2str(pred_gini_impurity)),file=f)
#        print('pred_margin_confidence=({})'.format(l2str(pred_margin_confidence)),file=f)
    
    d={'files':dicom_files,'labels':labels,'series_descriptions':series_descriptions,       'pred_gini_impurity':pred_gini_impurity,'pred_margin_confidence':pred_margin_confidence}
    
    with open('classification_output.csv',mode='w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=d.keys())
        w.writeheader()
        for row in zip(*d.values()): w.writerow(dict(zip(d.keys(),row)))
        
if __name__ == '__main__':
    main()
    
