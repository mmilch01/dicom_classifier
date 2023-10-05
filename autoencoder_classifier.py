# -*- coding: utf-8 -*-
import getpass, ipywidgets as ipw, os, json, shlex, io, re, tempfile, subprocess,unittest
import pydicom,numpy as np,csv,warnings,pickle,sys,tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from tensorflow import keras

from collections import Counter
import re

from IPython.display import FileLink
from matplotlib import pyplot as plt
from zipfile import ZipFile

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
#tf.logging.set_verbosity(tf.logging.ERROR)

#%load_ext autoreload
#%autoreload 2

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from juxnat_lib.xnat_utils import *

"""
Created on Mon Dec 12 15:45:06 2022

@author: mmilchenko
"""

class DICOMAutoencoderModel:
    def __init__(self):
        tags=[\
          'SeriesInstanceUID','Modality','Manufacturer','StudyDescription','SeriesDescription','ManufacturerModelName',\
            'BodyPartExamined','ScanningSequence','SequenceVariant','MRAcquisitionType',\
            'SequenceName','ScanOptions','SliceThickness','RepetitionTime','EchoTime','InversionTime',\
            'MagneticFieldStrength','NumberOfPhaseEncodingSteps','EchoTrainLength','PercentSampling',\
            'PercentPhaseFieldOfView','PixelBandwidth','AcquisitionMatrix','ImageType',\
            'FlipAngle','VariableFlipAngleFlag','PatientPosition','PhotometricInterpretation','Rows',\
            'Columns','PixelSpacing','ContrastBolusVolume','ContrastBolusTotalDose',\
            'ContrastBolusIngredient','ContrastBolusIngredientConcentration',\
            'PatientOrientation','ImageLaterality','ImageComments','ImagePositionPatient',\
            'ImageOrientationPatient','SamplesPerPixel',\
            'PlanarConfiguration','PixelAspectRatio','BitsAllocated','BitsStored','HighBit',\

            'PixelRepresentation','ColorSpace','AngioFlag','ImagingFrequency','EchoNumbers',\
            'SpacingBetweenSlices','TriggerTime','NominalInterval','BeatRejectionFlag','LowRRValue',\
            'HighRRValue','IntervalsAcquired','PVCRejection','SkipBeats','HeartRate','TriggerWindow',\
            'ReconstructionDiameter','ReceiveCoilName','TransmitCoilName','InPlanePhaseEncodingDirection',\
            'SAR','dBdt', 'B1rms', 'TemporalPositionIdentifier', 'NumberOfTemporalPositions', 'TemporalResolution',\
            'SliceProgressionDirection','IsocenterPosition', \
             \
             'KVP','DataCollectionDiameter','DistanceSourceToDetector','DistanceSourceToPatient',\
             'GantryDetectorTilt','TableHeight','RotationDirection','ExposureTime','XRayTubeCurrent','Exposure',\
             'ImageAndFluoroscopyAreaDoseProduct','FilterType','GeneratorPower','FocalSpots','ConvolutionKernel',\
             'WaterEquivalentDiameter','RevolutionTime','SingleCollimationWidth','TotalCollimationWidth',\
             'TableSpeed','TableFeedPerRotation','SpiralPitchFactor','DataCollectionCenterPatient',\
             'ReconstructionTargetCenterPatient','ExposureModulationType','EstimatedDoseSaving',\
             'CTDIvol','CalciumScoringMassFactorPatient','CalciumScoringMassFactorDevice','EnergyWeightingFactor',\
             'MultienergyCTAcquisition','AcquisitionNumber','RescaleIntercept','RescaleSlope',\
             'PatientSupportAngle','TableTopLongitudinalPosition','TableTopLateralPosition',\
             'TableTopPitchAngle','TableTopRollAngle',\
          \
          'StageName','StageNumber','NumberOfStages','ViewName','ViewNumber','NumberOfEventTimers',\
          'NumberOfViewsInStage','EventElapsedTimes','EventTimerNames','HeartRate','IVUSAcquisition','IVUSPullbackRate','IVUSGatedRate',\
          'TransducerType','FocusDepth','MechanicalIndex','BoneThermalIndex','CranialThermalIndex',\
          'SoftTissueThermalIndex','SoftTissueFocusThermalIndex','DepthOfScanField',\
          \
          'ExposureInuAs','AcquisitionDeviceProcessingDescription','AcquisitionDeviceProcessingCode',\
          'CassetteOrientation','CassetteSize','ExposuresOnPlate','RelativeXRayExposure','ExposureIndex',\
          'TargetExposureIndex','DeviationIndex','Sensitivity','PixelSpacingCalibrationType','PixelSpacingCalibrationDescription',\
          'DerivationDescription','AcquisitionDeviceProcessingDescription','AcquisitionDeviceProcessingCode',\
          'RescaleType','WindowCenterWidthExplanation','CalibrationImage','PresentationLUTShape',\
         \
          'PlateID','CassetteID','FieldOfViewShape','FieldOfViewDimensions','ImagerPixelSpacing',\
          'ExposureIndex','TargetExposureIndex','DeviationIndex','Sensitivity','DetectorConditionsNominalFlag',\
          'DetectorTemperature','DetectorType','DetectorConfiguration','DetectorDescription','DetectorMode',\
          'DetectorBinning','DetectorElementPhysicalSize','DetectorElementSpacing','DetectorActiveShape',\
          'DetectorActiveDimensions','DetectorActiveOrigin','DetectorManufacturerName','DetectorManufacturerModelName',\
          'FieldOfViewOrigin','FieldOfViewRotation','FieldOfViewHorizontalFlip','PixelSpacingCalibrationType',\
          'PixelSpacingCalibrationDescription',\
          \
          'PrimaryPromptsCountsAccumulated','SecondaryCountsAccumulated','SliceSensitivityFactor',\
          'DecayFactor','DoseCalibrationFactor','ScatterFractionFactor','DeadTimeFactor','IsocenterPosition',\
          'TriggerSourceOrType','CardiacFramingType','PVCRejection',\
          'CollimatorGridName','CollimatorType','CorrectedImage','TypeOfDetectorMotion','Units','CountsSource',\
          'ReprojectionMethod','SUVType','RandomsCorrectionMethod','RandomsCorrectionMethod','DecayCorrection',\
          'ReconstructionMethod','DetectorLinesOfResponseUsed','ScatterCorrectionMethod','ScatterCorrectionMethod',\
          'AxialMash','TransverseMash','CoincidenceWindowWidth','SecondaryCountsType',\
          \
          'PositionerType','PositionerPrimaryAngle','PositionerSecondaryAngle','PositionerPrimaryAngleDirection',\
          'ImageLaterality','BreastImplantPresent','PartialView','PartialViewDescription','OrganExposed']

        #code string
        self._tags_CS=['Modality', 'BodyPartExamined', 'MRAcquisitionType', 'VariableFlipAngleFlag',\
                       'PatientPosition','PhotometricInterpretation','ContrastBolusIngredient','ImageLaterality',\
                       'ColorSpace','AngioFlag','BeatRejectionFlag','InPlanePhaseEncodingDirection',\
                       'SliceProgressionDirection','RotationDirection','MultienergyCTAcquisition','IVUSAcquisition',\
                       'TransducerType','CassetteOrientation','CassetteSize','PixelSpacingCalibrationType',\
                       'CalibrationImage','PresentationLUTShape','FieldOfViewShape','DetectorConditionsNominalFlag',\
                       'DetectorConfiguration','DetectorActiveShape','FieldOfViewHorizontalFlip','PixelSpacingCalibrationType',\
                       'CollimatorType','TypeOfDetectorMotion','Units','CountsSource','ReprojectionMethod','SUVType',\
                       'RandomsCorrectionMethod','DecayCorrection','PositionerType','ImageLaterality','BreastImplantPresent',\
                       'PartialView','OrganExposed']

        #code string array, -1 signifies arbitrary length
        self._tags_CS_array={'ImageType': -1,'Patient Orientation': 2,'ExposureModulationType':-1,'CorrectedImage':-1,\
                             'SecondaryCountsType':-1,'ScanningSequence':-1,'SequenceVariant':-1}
        #long string (LO), long text (LT)
        self._tags_text=['Manufacturer', 'StudyDescription', 'SeriesDescription', 'ManufacturerModelName','ImageComments','PVCRejection',\
        'AcquisitionDeviceProcessingDescription','AcquisitionDeviceProcessingCode','PixelSpacingCalibrationDescription',\
        'DerivationDescription','AcquisitionDeviceProcessingDescription','AcquisitionDeviceProcessingCode','RescaleType',\
        'PlateID','CassetteID','DetectorDescription','DetectorMode','DetectorManufacturerName','DetectorManufacturerModelName',\
        'PixelSpacingCalibrationDescription','TriggerSourceOrType','CardiacFramingType','PVCRejection','ReconstructionMethod',\
        'DetectorLinesOfResponseUsed','ScatterCorrectionMethod','PartialViewDescription']

        #short string (may be multi-word)
        self._tags_SH=['SequenceName','ReceiveCoilName','TransmitCoilName','FilterType','StageName','ViewName',\
                       'CollimatorGridName']
        #string array (SH array, LO array)
        self._tags_array_SH={'ConvolutionKernel':-1,'EventTimerNames':-1,'WindowCenterWidthExplanation':-1,'ScanOptions':-1}
        #decimal (float) string (DS,FL,FD)
        self._tags_float=['SliceThickness','RepetitionTime','EchoTime','InversionTime','MagneticFieldStrength','PercentSampling',\
                       'PercentPhaseFieldOfView','PixelBandwidth','FlipAngle','ContrastBolusVolume','ContrastBolusTotalDose',\
                       'ContrastBolusIngredientConcentration','ImagingFrequency','SpacingBetweenSlices','TriggerTime',\
                       'ReconstructionDiameter','SAR','B1rms','TemporalResolution','IsocenterPosition','KVP','DataCollectionDiameter',\
                       'DistanceSourceToDetector','DistanceSourceToPatient','GantryDetectorTilt','TableHeight',\
                       'ImageAndFluoroscopyAreaDoseProduct','WaterEquivalentDiameter','RevolutionTime','TotalCollimationWidth',\
                       'SingleCollimationWidth','TableSpeed','TableFeedPerRotation','SpiralPitchFactor','EstimatedDoseSaving',\
                       'CTDIvol','CalciumScoringMassFactorPatient','EnergyWeightingFactor','RescaleIntercept','RescaleSlope',\
                       'PatientSupportAngle','TableTopLongitudinalPosition','TableTopLateralPosition','TableTopPitchAngle',\
                       'TableTopRollAngle','IVUSPullbackRate','IVUSGatedRate','FocusDepth','MechanicalIndex','BoneThermalIndex',\
                       'CranialThermalIndex','SoftTissueThermalIndex','SoftTissueFocusThermalIndex','ExposureIndex',\
                       'TargetExposureIndex','DeviationIndex','Sensitivity','DeviationIndex','DetectorTemperature',\
                       'FieldOfViewRotation','SliceSensitivityFactor','DecayFactor','DoseCalibrationFactor','ScatterFractionFactor',\
                       'DeadTimeFactor','CoincidenceWindowWidth','PositionerPrimaryAngle','PositionerSecondaryAngle',\
                       'PositionerPrimaryAngleDirection']


        #integer string (IS,US)
        self._tags_integer=['NumberOfPhaseEncodingSteps','EchoTrainLength','PercentSampling','Rows','Columns','SamplesPerPixel',\
                            'PlanarConfiguration','BitsAllocated','BitsStored','HighBit','PixelRepresentation','NominalInterval',\
                            'LowRRValue','HighRRValue','IntervalsAcquired','SkipBeats','TriggerWindow',\
                            'TemporalPositionIdentifier','NumberOfTemporalPositions','ExposureTime','XRayTubeCurrent',\
                            'Exposure','GeneratorPower','AcquisitionNumber','StageNumber','NumberOfStages','ViewNumber',\
                            'NumberOfEventTimers','NumberOfViewsInStage','DepthOfScanField','ExposureInuAs','ExposuresOnPlate',\
                            'RelativeXRayExposure','ExposureIndex','TargetExposureIndex','PrimaryPromptsCountsAccumulated',\
                            'TransverseMash']

        #integer arrays (IS,US)
        self._tags_array_int={'AcquisitionMatrix':4,'PixelAspectRatio':2,'EchoNumbers':-1,'FieldOfViewDimensions':-1,\
                              'AxialMash':2}
        #float arrays (DS,FD)
        self._tags_array_float={'PixelSpacing':2,'ImagePositionPatient':3,'ImageOrientationPatient':6,'FocalSpots':-1,\
                                'DataCollectionCenterPatient':3,'ReconstructionTargetCenterPatient':3,\
                                'CalciumScoringMassFactorDevice':3,'EventElapsedTimes':-1,'ImagerPixelSpacing':2,\
                                'DetectorBinning':2,'DetectorElementPhysicalSize':2,'DetectorElementSpacing':2,\
                                'DetectorActiveDimensions':-1,'DetectorActiveOrigin':2,'FieldOfViewOrigin':2,\
                                'SecondaryCountsAccumulated':-1,'IsocenterPosition':3}               

        self._string_tags=self._tags_CS+self._tags_text+self._tags_SH
        self._string_array_tags_dict={**self._tags_CS_array,**self._tags_array_SH}
        self._string_array_tags=list(self._tags_CS_array.keys())+list(self._tags_array_SH.keys())

        self._num_tags=self._tags_float+self._tags_integer
        self._num_array_tags_dict={**self._tags_array_int, **self._tags_array_float}
        self._num_array_tags=list(self._tags_array_int.keys())+list(self._tags_array_float.keys())

        self._all_tags=list(set(tags))
        self._all_tags.sort()

        self.vectorizer,self.autoencoder_model=None,None

    def convert_to_builtin(self,value):
        # If the value is a dictionary, recursively convert its items
        if isinstance(value, dict):
            return {k: convert_to_builtin(v) for k, v in value.items()}
    
        # If the value is a list or MultiValue, convert its elements
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            return [convert_to_builtin(v) for v in value]
    
        # Convert pydicom specific types
        if isinstance(value, pydicom.uid.UID):
            return str(value)
        if isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
            return float(value)
    
        # Return the value unchanged if it's already a built-in type
        return value
    
    def convert_scans_to_builtin(self,d):
        return {k: self.convert_to_builtin(v) for k, v in d.items()}
    
    def write_scans_json(self,scans,file):
        scans_new=[ self.convert_scans_to_builtin(s) for s in scans ]
        with open(file, "w") as fp:
            json.dump(scans,fp)
    
    '''
        get some useful statistics from the dataset.
    '''
    def analyze_scans(self, scans):
        max_lengths = {}
        for tag in self._num_array_tags:
            max_length = max([len(scan.get(tag)) if isinstance(scan.get(tag), (list, tuple)) else 1 for scan in scans])
            max_lengths[tag] = max_length
        return max_lengths

    def scans_from_files(self,file_list,tags=None):
        if tags is None: tags=self._all_tags
        scans=[]
        for f in file_list:
            d=dict()
            ds=pydicom.filereader.dcmread(file,stop_before_pixels=True,specific_tags=tags)            
            for tag in tags:
            try:
                d[tag]=ds[tag].value
            except Exception as e:
                pass
            scans+=[d]
        return scans
        
    def generate_scanlist(self,input_dir,out_file=None, tags=None):
        '''
        Generates scan list from the directory populated by DICOM files.
        Writes pickled dictionary file.
        '''
        if tags is None: tags=self._all_tags
        scans=[]
        for root, dirs, files in os.walk(input_dir):
            if not files: break
            i=0
            for f in files:
                d=dict()
                file=os.path.join(root,f)
                ds=pydicom.filereader.dcmread(file,stop_before_pixels=True,specific_tags=tags)            
                if not i % 10000:
                    print ('reading file {}'.format(i))
                for tag in tags:
                    try:
                        d[tag]=ds[tag].value
                    except Exception as e:
                        pass
                i+=1
                #if i>max_it: break
                scans+=[d]
    #            if not i % max_scans:
    #                n+=1
                    #write next scans file.
    #                write_pkl_scans(d,out_file+'_'+str(n)+'.pkl')
    #                d={tag:[] for tag in tags}                                
    #    write last file
     #   n+=1
        if out_file is not None:
            self.write_pkl_scans(scans,out_file+'.pkl')
        return scans

    def write_pkl(self,object,file):        
        with open(file, 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        print('written',file)
        
    def read_pkl(self,file):
        with open(file, 'rb') as handle:
            object=pickle.load(handle)
        print('read', file)
        return object
        
    def write_pkl_scans(self,d,file):
        print ('writing file {}'.format(file))                
        with open(file, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)               
            
    def read_pkl_scans(self, file):
        print('reading file {}'.format(file))
        with open(file,'rb') as f:
            self._scans=pickle.load(f)
        return self._scans
    
    def init_and_run_nn_training(self,scans,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        '''
        Inits and runs NN training from the given set of scans.
        '''
        #print('Checking labeling validity...')
        #if not self.labeling_valid(scans):
        #    print('Invalid labeling, cannot train. Either fix labels or remove unlabeled records from the training set.')
        #    return False
        
        print('Generating vocabulary...')
        self.gen_vocabulary(scans)

        print('Preparing training vectors...')
        descs,y=self.prepare_training_vectors_nn(scans)
        print('Training...')
        self.train_nn(descs,y,test_split=test_split,epochs=epochs,\
                      batch_size=batch_size,random_state=random_state)
        print('Done.')
        return True
                    
    def prepare_descs(self,scans):
        #descs are 'sentences' that contain all supported tags and xnat fields.
        #(former series description and log-compressed number of frames.)
        return [self.prepare_desc(s) for s in scans ]
                         
    def prepare_desc(self,scan):
        s=scan
        text=[]              
        for key, vals in s.items():
            if key in self._string_tags:
                try:
                    text+=["{}_{}".format(key,val) for val in vals.split(" ") if len(val)>0 ]
                except Exception as e:
                    print('WARNING: value error for key {}, value {}'.format(key,vals))
                    #raise ValueError("key: {}, value: {}".format(key,vals))
                    
            elif key in self._string_array_tags:
                try:
                    text+=[ "{}{}_{}".format(key,i,vals[i]) for i in range(len(vals)) if len(vals[i])>0 ]
                except Exception as e:
                    print('WARNING: value error for key: {}, value: {}'.format(key,vals))
                    #raise ValueError("key: {}, value: {}".format(key,vals))
                         
        return ' '.join([w for w in text ]) #if ((not w.isdigit()) and (len(w)>1))

    def serialize_vectorizer(self,file_root,is_load):
        if is_load:
            try: 
                # Load config
                with open(file_root+'.json', 'r') as f:
                    loaded_config = json.load(f)            
                # Create new layer with loaded config
                self.vectorizer = TextVectorization.from_config(loaded_config)            
                # Load weights
                with open(file_root+'.vec', 'rb') as f:                
                    loaded_weights = pickle.load(f)
                # Set weights
                self.vectorizer.set_weights(loaded_weights)
                print('Vocabulary loaded: ',file_root)
            except Exception as e:
                print(e)
                
        else:
            try:
                weights = self.vectorizer.get_weights()
                config = self.vectorizer.get_config()
                # Save weights
                with open(file_root+'.vec', 'wb') as f:
                    pickle.dump(weights, f)        
                # Save config
                with open(file_root+'.json', 'w') as f:
                    json.dump(config, f)
                print('Vocabulary saved: ', file_root)
            except Exception as e: 
                print(e)
                

    def filter_least_frequent_words(self, descs, max_unique_words):
        """
        Filters out the least frequent words from the list of descriptions to match the maximum count
        of unique words specified by max_unique_words.
        Parameters:
            descs (list): List of descriptions (strings) to be filtered.
            max_unique_words (int): Maximum number of unique words to keep.        
        Returns:
            list: List of filtered descriptions.
        """
        
        # Tokenize the descriptions to get individual words
        all_words = []
        for desc in descs:
            words = desc.lower().split(' ')
            all_words.extend(words)
            
        # Count the frequency of each word
        word_freq = Counter(all_words)
            
        # Get the top max_unique_words based on their frequency
        most_common_words = set(word for word, _ in word_freq.most_common(max_unique_words))    
    
        # Filter the descriptions
        filtered_descs = []
        for desc in descs:
            words = desc.lower().split(" ")
            filtered_words = [word for word in words if word in most_common_words]
            if filtered_words:
                filtered_desc = ' '.join(filtered_words)
                filtered_descs.append(filtered_desc)            
        return filtered_descs
            
    def gen_vocabulary(self,scans,max_tokens=20000):        
        print('generating vocabulary...')
        descs=self.filter_least_frequent_words(self.prepare_descs(scans),max_tokens)
        ds=tf.data.Dataset.from_tensor_slices(descs)
        vectorizer=TextVectorization(max_tokens=max_tokens,output_mode="multi_hot")
        vectorizer.adapt(ds)
        self.vectorizer=vectorizer
        print('the length of vocabulary is ',len(vectorizer.get_vocabulary()))    
    
    def multi_hot_to_text(self, multi_hot_tensor, inverse_vocab):
        """
        Convert a multi-hot encoded tensor to its original text representation.
    
        Parameters:
        - multi_hot_tensor: A multi-hot encoded tensor.
        - inverse_vocab: A dictionary mapping from index to token.
    
        Returns:
        A list of strings where each string is the original text representation.

        Example:
        inverse_vocab=dict(enumerate(daem.vectorizer.get_vocabulary()))
        decoded=multi_hot_to_text(ds[:10],inverse_vocab)
        decoded[0]
        """
        
        text_representation = []
        for sample in multi_hot_tensor:
            indices = tf.where(sample).numpy().flatten()
            tokens = [inverse_vocab[idx] for idx in indices]
            original_text = ' '.join(tokens)
            text_representation.append(original_text)
        return text_representation

    def get_encoder(self,inputs,dim0=256):
        #1000, 64, input_length1000, 64, input_length=10=10
        x=layers.Dense(dim0,activation="relu",name='encoder_input')(inputs)
        #x=layers.Dropout(.5)(x)
        x=layers.Dense(dim0/2,activation="relu")(x)
        x=layers.Dense(dim0/4,activation="relu")(x)
        x=layers.Dense(dim0/8,activation="relu",name='encoder_output')(x)
        return x

    def get_encoder_embedding(self,inputs,dim0=256):
        #1000, 64, input_length1000, 64, input_length=10=10
        #embedding_layer=Embedding(inputs.shape[0],50,input_length=80,padding='post')

            
        x=layers.Dense(dim0,activation="relu",name='encoder_input')(inputs)
        #x=layers.Dropout(.5)(x)
        x=layers.Dense(dim0/2,activation="relu")(x)
        x=layers.Dense(dim0/4,activation="relu")(x)
        x=layers.Dense(dim0/8,activation="relu",name='encoder_output')(x)
        return x


    
    def get_decoder(self,encoded, max_tokens=10000,dim0=256):
        #x=layers.Dropout(.5)(x)    
        x=layers.Dense(dim0/4,activation="relu",name='decoder_input')(encoded)
        x=layers.Dense(dim0/2,activation="relu")(x)
        x=layers.Dense(dim0,activation="relu")(x)
        outputs=layers.Dense(max_tokens, activation='sigmoid', name='decoder_output')(x)
        return outputs


    
    def get_model(self,max_tokens=10000,base_layer_size=256):
        dim0=base_layer_size
        inputs=keras.Input(shape=(max_tokens,))
        encoded_inputs=self.get_encoder(inputs,dim0)
        outputs=self.get_decoder(encoded_inputs,inputs.shape[1],dim0)
        
        model=keras.Model(inputs,outputs)
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        optimizer.learning_rate.assign(0.001)
        model.compile(optimizer,loss="mean_squared_error",metrics=["MeanSquaredError"])
        #model.compile(optimizer="adam",loss="mean_squared_error",metrics=["MeanSquaredError"])
        return model

    def load_model_autoencoder(self,zipfile):
        try:
            zipfile_dir=os.path.dirname(zipfile)
            print('extracting',zipfile)
            with ZipFile(zipfile,'r') as zf:
                zf.extractall(zipfile_dir)
            rt=os.path.splitext(zipfile)[0]
            vec_file,json_file,model_file=rt+'.vec',rt+'.json',rt+'.h5'
            print('loading vocabulary from',rt)
            self.serialize_vectorizer(rt,True)
            print('loading mdoel from',model_file)
            self.autoencoder_model=tf.keras.models.load_model(model_file)
        except Exception as e:
            print('error loading: '+zipfile)
            print(e)
        os.remove(vec_file)
        os.remove(model_file)
        os.remove(json_file)
        
        return self.vectorizer is not None and self.autoencoder_model is not None

    def save_model_autoencoder(self, zipfile):
        if self.autoencoder_model is None or self.vectorizer is None: return False

        try: 
            zipfile_dir=os.path.dirname(zipfile)
            rt=os.path.splitext(zipfile)[0]
            zf,vecf,jsf,hd5f=rt+'.zip',rt+'.vec',rt+'.json',rt+'.h5'
    
            #save individual files for vectorizer and model
            print('saving vectorizer')
            self.serialize_vectorizer(rt,False)
            print('saving model')
            self.autoencoder_model.save(hd5f)
            print('zipping files')
            with ZipFile(zf,'w') as zipfile: 
                zipfile.write(hd5f,arcname=os.path.basename(hd5f))
                zipfile.write(vecf,arcname=os.path.basename(vecf))
                zipfile.write(jsf,arcname=os.path.basename(jsf))
        except Exception as e:
            print('error saving:',zipfile)
            print(e)
            
        os.remove(vecf)
        os.remove(jsf)
        os.remove(hd5f)

    def train_autoencoder(self, descs, checkpoint_dir='./autoencoder_training', epochs=100, batch_size=128, base_layer_size=256, learning_rate=0.001):
        #feeder function for dataset mapping.        
        def vectorize_text(text):
            return self.vectorizer(text), self.vectorizer(text)
        if self.vectorizer is None: 
            print ('Training cannot proceed: vocabulary is not defined')
            return False
        
        #create a tf.data.Dataset object, to be used for training.
        ds = tf.data.Dataset.from_tensor_slices(descs)
        ds = ds.batch(batch_size)  # Define your batch size here
        ds = ds.map(vectorize_text)
        dict_size=len(self.vectorizer.get_vocabulary())

        if self.autoencoder_model is None:
            self.autoencoder_model=self.get_model(dict_size,base_layer_size)
        else:
            print('Continuing training on existing model')
                    
        self.autoencoder_model.summary()
        model.optimizer.learning_rate.assign(learning_rate)
        model.fit(ds,epochs=epochs, callbacks=[keras.callbacks.ModelCheckpoint(filepath=check,save_best_only=False,monitor='mean_square_error')])                
    
    def get_classifier_model(self,original_model, num_classes, max_tokens=10000,hidden_layer_size=256):
        encoder_output = original_model.get_layer('encoder_output').output
        inputs=keras.Input(shape=(max_tokens,))
        hidden_layer = layers.Dense(hidden_layer_size/4, activation='relu')(encoder_output)
        classifier_output = layers.Dense(num_classes, activation='softmax')(hidden_layer)
        #classifier_output = layers.Dense(num_classes, activation='softmax')(encoder_output)
        classifier_model = keras.Model(original_model.input, classifier_output)
    
        #freeze the encoder layers.
        encoder_end_index=len(classifier_model.layers)-3
        #for layer in classifier_model.layers[:encoder_end_index]: layer.trainable=False
            
        classifier_model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["accuracy","categorical_accuracy"])
        return classifier_model
        
class AutoencoderClassifierTest:
    def __init__(self,daem:DICOMAutoencoderModel):
        self.daem=DICOMAutoencoderModel()
        self.vectorizer=vectorizer
    
    def test_load_scans1(self,scans):
        self.scans=self.daem.read_pkl_scans('/data/ImagingCommons/data.ImagingCommons.416237_1.pkl')        

    def test_load_model1(self,scans):
        self.daem.load_model_autoencoder('test/autoencoder.256.20K_tokens.09252023.zip')

    def test_prepare_scans1(self,scans):
        self.descs=self.daem.prepare_descs(scans)
    
    def test_load_model1(self):
        self.descs=self.daem.prepare_descs(scans)
        
    def test_load_nomenclature2(self):
        print('loading from file:',self.scm.load_from_file('./test/mri_types.json')==True)
        
            
    def test_load_training_set1(self):
        try:
            self.scans=self.usc.read_scans_csv('./test/all_scans_hofid.csv')
            if len(self.scans)>0: 
                print ('loaded scans from','./test/all_scans_hofid.csv')
        except Exception as e:
            print(e)
            print('loading scans from file failed')
            
    def test_load_training_set2(self):
        self.scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
            
    def test_train_model1_nn(self,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        #try:      
        self.test_load_nomenclature1()
        self.test_load_training_set1()
        self.usc.init_and_run_nn_training(self.scans,test_split=test_split,\
                                          epochs=epochs,batch_size=batch_size,random_state=random_state)
        
        self.usc.save_model_nn('./test/neuro-onc-test.zip')
        #except Exception as e:
        #    print('Exception:',e)
        
    def test_train_model1_svm(self,test_split=0.11,random_state=1000):
        self.test_load_nomenclature1()
        self.test_load_training_set1()
        self.usc.init_and_run_svm_training(self.scans,test_split=test_split,random_state=random_state)        
        self.usc.save_model('./test/neuro-onc-test_svm.pkl')
        
    def test_train_model2_svm(self,test_split=0.11,random_state=1000):
        self.test_load_nomenclature2()
        self.test_load_training_set2()
        self.usc.init_and_run_svm_training(self.scans,test_split=test_split,random_state=random_state)        
        self.usc.save_model('./test/mri_types-test_svm.pkl')
        
    def test_train_model2_nn(self,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        self.test_load_nomenclature2()
        self.test_load_training_set2()
        self.scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
        print ('loaded scans from','./test/all_scans_function.csv')
        self.usc.init_and_run_nn_training(self.scans,test_split=test_split,\
                                          epochs=epochs,batch_size=batch_size,random_state=random_state)
        self.usc.save_model_nn('./test/mri_types-test.zip')
        
    def prediction_accuracy(self,labeled_scans,classified_types):
        scans=labeled_scans
        n=0
        for i in range(len(scans)):
            if classified_types[i]!=scans[i]['hof_id']:
                print('position: {}, predicted: {}, actual: {}'\
                      .format(i,classified_types[i],scans[i]['hof_id']))
                n+=1
        print('Classification accuracy:',1.-n/len(scans))
        print("Done.")
        
    
    def test_validate_model1_svm(self):
        self.test_load_nomenclature1()
        self.usc.load_model('./test/neuro-onc-test_svm.pkl')
        scans=self.usc.read_scans_csv('./test/all_scans_hofid.csv')
        classified_types=self.usc.infer_svm(scans)
        self.prediction_accuracy(scans,classified_types)
        
    def test_validate_model2_svm(self):
        self.test_load_nomenclature2()
        self.usc.load_model('./test/mri_types-test_svm.pkl')
        scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
        classified_types=self.usc.infer_svm(scans)
        self.prediction_accuracy(scans,classified_types)        
                            
    def test_validate_model1_nn(self):
        self.test_load_nomenclature1()
        self.usc.load_model_nn('./test/neuro-onc-test.zip')
        scans=self.usc.read_scans_csv('./test/all_scans_hofid.csv')
        classified_types=self.usc.infer_nn(scans)
        self.prediction_accuracy(scans,classified_types)
            
    def test_validate_model2_nn(self):
        self.test_load_nomenclature2()
        self.usc.load_model_nn('./test/mri_types-test.zip')
        scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
        classified_types=self.usc.infer_nn(scans)
        self.prediction_accuracy(scans,classified_types)

    def test_infer_model2_svm(self):
        self.usc.load_model('./test/mri_types-test_svm.pkl')
        scans=self.usc.read_scans_csv('./test/all_scans.csv')
        classified_scans=self.usc.predict_classifier(scans)
        self.usc.write_scans_csv(classified_scans,'./test/all_scans-mri_types_predicted_svm.csv')
        
            
       
                             
    