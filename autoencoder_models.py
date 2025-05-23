#!/usr/bin/env python
# coding: utf-8
import csv, getpass, io, json, os, pickle, pydicom, re, shlex, subprocess, sys, tempfile, tensorflow as tf, unittest, warnings, zipfile, argparse, numpy as np, math,sklearn, pandas as pd

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

from juxnat_lib import *
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

    def create_classifier_model(self,scan_types,trainable_embedding=False):
        embedding_layer = self.autoencoder.layers[1]
        embedding_layer.trainable = trainable_embedding
        self.max_length=embedding_layer.input_shape[1]

        input_layer = Input(shape=(self.max_length,))
        embedding_output = embedding_layer(input_layer)

        attention_output = MultiHeadAttention(num_heads=2,key_dim=50)(embedding_output,embedding_output)
        encoder_output=Dense(50, activation="relu")(attention_output)
        flattened_output=Flatten()(encoder_output)
        dense1=Dense(100,activation='relu')(flattened_output)
        x=Dropout(0.5)(dense1)
        output_layer = Dense(len(scan_types), activation='softmax')(x)
        #output_layer = Dense(categorical_labels.shape[1], activation='softmax')(x)

        self.classification_model = Model(inputs=input_layer, outputs=output_layer)
        self.classification_model.summary()        
        self.classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.classification_model

    def train_classifier_model(self,X1_train,Y1_train,epochs=100,batch_size=512,validation_split=0.2):
        self.classification_model.optimizer.learning_rate.assign(0.001)
        reducer = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=10)
        self.classification_model.fit(X1_train, Y1_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks=[reducer])

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
            
    def save_model(self, model, root):
        try: 
            tokenizer_file,h5file=root+'.tokenizer',root+'.h5'    
            #save individual files for vectorizer and model
            print('saving tokenizer to',tokenizer_file)
            self.daem.write_pkl(self.tokenizer,tokenizer_file)        
            print('saving model to',h5file)
            model.save(h5file)            
        except Exception as e:
            print('error saving:',h5file)
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

    def classify_dicom_scans(self, scans, tokenizer_file, model_file, nomenclature_file,confidence_level=0.95,unknown_label='UNKNOWN',verbose=0):        
        self.tokenizer,self.classification_model=self.load_model(tokenizer_file,model_file)
        max_length=self.classification_model.layers[0].input_shape[0][1]
        sd=self.sequences=self.sequences_from_scans(scans,max_length)
        #print(self.sequences.shape)
        #print(self.sequences)
        
        scan_types=self.load_nomenclature(nomenclature_file)
        pred=self.classification_model.predict(self.sequences)

        max_len=min(10,len(pred))
        if verbose>1: print('pred (first 10):',pred[:max_len])

        #pred_ord=np.argmax(pred,1)
        pred_ord=np.argsort(-pred,axis=1)

        #print('pred_ord:',pred_ord)

        label_encoder = LabelEncoder()
        label_encoder.fit_transform(self.load_nomenclature(nomenclature_file))                                    
        pred_inv0=label_encoder.inverse_transform(pred_ord[:,0])
        if verbose>1: print('pred_inv0 (first 10):',pred_inv0[:max_len])
        pred_inv1=label_encoder.inverse_transform(pred_ord[:,1])
        if verbose>0: print('pred_inv1 (first 10):',pred_inv1[:max_len])
        #out_label=[]
        series_descriptions=[]
        #prediction quality metrics
        #predicted most likely class
        pred_class1=[]
        pred_prob1=[]
        #predicted second most likely class
        pred_prob2=[]
        pred_class2=[]

        pred_entropy=[]
        pred_gini_impurity=[]
        pred_margin_confidence=[]
        
        for i in range (0,len(pred_inv0)):
            pred_cur=pred[i]
            pred_ord_cur=pred_ord[i]
            #max_pred=sorted_args[0]
            #max_pred=pred_ord[i,0]
            #max_pred=np.argmax(pred_cur)
            #out_label+=[pred_inv0[i]]
            #if pred_cur[max_pred]>=confidence_level:
            #    out_label+=[pred_inv0[i]]
            #else:
            #    out_label+=[unknown_label]

            pred_class1+=[ pred_inv0[i] ]
            pred_prob1+=[ pred_cur[pred_ord_cur[0]] ]
            pred_class2+=[ pred_inv1[i] ]
            pred_prob2+=[ pred_cur[pred_ord_cur[1]] ]

            pred_gini_impurity+=[1-np.sum(np.array([pred_cur[i]*pred_cur[i] for i in range (0,len(pred_cur))]))]            
            pred_margin_confidence+=[pred_cur[pred_ord_cur[0]]-pred_cur[pred_ord_cur[1]]]

            try:
                series_descriptions+=[scans[i]['SeriesDescription'].replace(' ','_')+' ']
            except Exception as e:
                series_descriptions+=['NA']
                if verbose>0: print('no series description for file',i)
        if verbose>0: print('Predicted labels (first 10):',pred_class1[:max_len])
        return pred_class1,pred_prob1,pred_class2,pred_prob2,pred_gini_impurity,pred_margin_confidence,series_descriptions
        
class AttentionModelTest: 
    def __init__(self):
        daem=DICOMAutoencoderModel()
        self.am=AttentionModel()

    def train_model1(self):
        am,daem=self.am,self.am.daem
        #input
        tokenizer_file='./model_mirrir_1351062s_15Kt.10.04.2023/tokenizer.pkl'        
        label_file='./test/compare/manual_label_based_on_classification_output_model_fc_39374-600.03.20.2024_2024Apr06_113650.csv'
        ae_model_file='autoencoder.dualhead.21ep.h5'
        ae_model_dir='./model_mirrir_1351062s_15Kt.10.04.2023/'
        nomenclature_file='./model_mirrir_1351062s_15Kt.10.04.2023/neuro_onc.json'
        
        #output
        classifier_model_root='./test/classifier.neuro-onc_mirrir1351062_04112024'        
        df=pd.read_csv(label_file)
        dicom_files=df['files']
        print('Reading scans')
        scans=am.daem.scans_from_files(dicom_files)
        am.load_autoencoder(ae_model_dir,model_file=ae_model_file)
        scan_types=am.load_nomenclature(nomenclature_file)
        am.create_classifier_model(scan_types,trainable_embedding=False)
        am.tokenizer=am.daem.read_pkl(tokenizer_file)
        print('Creating sequences')
        seqs=am.sequences_from_scans(scans,am.max_length)
        X1 = pad_sequences(seqs, maxlen=77, padding='post')
        labels=df['hof_id']
        
        le=LabelEncoder()
        le.fit_transform(scan_types)
        encoded_labels=le.fit_transform(labels)        
        categorical_labels = to_categorical(encoded_labels)
        
        X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, categorical_labels, test_size=0.1, random_state=42)
        am.train_classifier_model(X1_train,Y1_train,epochs=100,batch_size=512,validation_split=0.1)
        am.evaluate_classifier_model(X1,categorical_labels)
        am.save_model(am.classification_model,classifier_model_root)
        print('Done')

    def evaluate_model1(self):
        am,daem=self.am,self.am.daem
        label_files=['./test/compare/tcga_classification_output_model_fc_comparison.xlsx',
               './test/compare/M19021_glioma2_classification_output_model_fc_mirrir_compare.xlsx',
               './test/compare/manual_label_based_on_classification_output_model_fc_39374-600.03.20.2024_2024Apr06_113650.xlsx']

        dataset_labels=['TCGA-Glioma', 'M19021_glioma2','CONDR_and_CONDR_METS']
        tokenizer_file='./model_mirrir_1351062s_15Kt.10.04.2023/tokenizer.pkl'
        model_file='./test/classifier.neuro-onc_mirrir1351062_04112024.h5'
        #model_file='./model_mirrir_1351062_36530.04.07.2024/classifier.neuro-onc_mirrir1351062_04072024.h5'
        nomenclature_file='./model_mirrir_1351062s_15Kt.10.04.2023/neuro_onc.json'
        
        for file,dataset in zip(label_files,dataset_labels):
            print(f'Evaluating {model_file} on {dataset}')
            df=pd.read_excel(file)
            dicom_files=df['files']
            sdescs=df['SeriesDescription']
            print('Reading DICOM files')
            scans=am.daem.scans_from_files(dicom_files)
            manual_labels=df['labels_manual']
            print('running classification of {} scans.'.format(len(scans)))
            labels1,probs1,labels2,probs2,pred_gini_impurity,pred_margin_confidence,series_descriptions=am.classify_dicom_scans(scans, tokenizer_file, model_file, nomenclature_file,confidence_level=0.5,verbose=1)
            nMatch,nMismatch=0,0
            for sd,manual_label,computed_label in zip (sdescs,manual_labels,labels1):
                if manual_label == computed_label: 
                    nMatch+=1
                else: 
                    nMismatch+=1
                    if nMismatch<10:
                        print(f'description: {sd}, manual label: {manual_label}, computed label: {computed_label}')
            print(f'Total labels: {nMatch+nMismatch}, correct: {nMatch}, incorrect: {nMismatch}, accuracy: {nMatch/(nMatch+nMismatch)}')
            break

def parse_paths(paths, path_type):
    '''
    extract scan and experiment ID's from file paths, to put in the output csv.
    '''
    experiments,scans=[],[]
    for path in paths:
        experiment_id = 'NA'
        scan_id = 'NA'
        
        if path_type == 'project':
            match = re.match(r'.*/([^/]+)/([^/]+)/SCANS/([^/]+)/DICOM/([^/]+)', path)
            if match:
                experiment_id = match.group(2)
                scan_id = match.group(3)
        elif path_type == 'experiment':
            match = re.match(r'.*/SCANS/([^/]+)/DICOM/([^/]+)', path)
            if match:
                scan_id = match.group(1)
                
        experiments+=[experiment_id]
        scans+=[scan_id]
        
    return experiments,scans

def parse_args():
    parser = argparse.ArgumentParser(description='Classify a list of DICOM files using a trained model.')
    #parser.add_argument('dicom_files', type=str, nargs='+', help='List of paths to DICOM files to be classified.')
    parser.add_argument('--file_list', type=str, help='file with input DICOM file list', required=True)
    parser.add_argument('--model_file', type=str, help='trained model file',required=True)
    parser.add_argument('--tokenizer_file', type=str, help='tokenizer file',required=True)
    parser.add_argument('--nomenclature_file', type=str, help='nomenclature file',required=True)
    parser.add_argument('--path_type', type=str, help='XNAT path type (scan,experiment,project)',required=True)
    parser.add_argument('--tag_out', type=str, action='append', help='optional DICOM tag (string name, can be repeated) to output in csv',required=False)
    return parser.parse_args()


def l2str(arr):
    lst=[str(arr[i]) for i in range(0,len(arr))]
    return ' '.join(lst)

def main():
    args = parse_args()
    #dicom_files = args.dicom_files
    dicom_files=[]
    file_list=args.file_list
    model_file = args.model_file
    tokenizer_file = args.tokenizer_file
    nomenclature_file = args.nomenclature_file
    tags_out=args.tag_out
    path_type=args.path_type
    
    # Verify that the specified DICOM files exist
    if not os.path.exists(file_list):    
        print("Error: input file list {} does not exist".format(file_list))
        sys.exit(1)
    
    with open(file_list, 'r') as file:
        for line in file:
            dicom_files.append(line.strip())

    for file in dicom_files:
        if not os.path.exists(file):
            print(f"Error: Specified DICOM file does not exist: {file}", file=sys.stderr)
            sys.exit(1)
            
    am=AttentionModel()
    print('reading scans from DICOM')
    scans=am.daem.scans_from_files(dicom_files)
    print('running classification of {} scans.'.format(len(scans)))
    labels1,probs1,labels2,probs2,pred_gini_impurity,pred_margin_confidence,series_descriptions=am.classify_dicom_scans(scans, tokenizer_file, model_file, nomenclature_file,confidence_level=0.5)
        
    d={'files':dicom_files,'labels1':labels1,'probs1':probs1,'labels2':labels2,'probs2':probs2,'series_descriptions':series_descriptions, 'pred_gini_impurity':pred_gini_impurity,'pred_margin_confidence':pred_margin_confidence}

    #add experiment and scan columns.    
    d['experiment'],d['scan']=parse_paths(dicom_files,path_type)
    
    #append requested DICOM tags.
    for tag in tags_out:
        d[tag]=[ s[tag] if tag in s.keys() else 'NA' for s in scans ]
    
    with open('classification_output.csv',mode='w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=d.keys())
        w.writeheader()
        for row in zip(*d.values()): w.writerow(dict(zip(d.keys(),row)))
        
if __name__ == '__main__':
    main()
    
