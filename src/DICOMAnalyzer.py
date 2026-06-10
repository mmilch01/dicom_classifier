import math

class DICOMAnalyzer:
    def __init__(self):
        tags=[\          'SeriesInstanceUID','Modality','Manufacturer','StudyDescription','SeriesDescription','ManufacturerModelName',\
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

        #number of bins to sample numeric tags into.
        self.numeric_tag_bins=10        

    def scans_from_files(self,file_list,tags=None):
        '''
        Returns a list of scans, represented as dictionaries over specified tags, for a file list
        '''
        if tags is None: tags=self._all_tags
        scans=[]
        for file in file_list:
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
                scans+=[d]
        if out_file is not None:
            self.write_pkl_scans(scans,out_file+'.pkl')
        return scans

    def get_dataset_stats(self,scans):
        '''
            stats is dict of the form: {<tag1>:[Min Max], <tag2>:[Min Max], ...}
        '''        
        stats={}
        for scan in scans:
            for key,val in s.items():
                if key in self._num_tags:
                    if key not in stats:
                        stats[key]=[val,val]
                    else:
                        vmin,vmax=stats[key][0],stats[key][1]
                        stats[key]=[min(val,vmin),max(val,vmax)]
        return stats

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

    def prepare_descs(self,scans):
        #descs are 'sentences' that contain all supported tags and xnat fields.
        #(former series description and log-compressed number of frames.)
        stats=self.get_dataset_stats(scans)
        return [self.prepare_desc(s,stats) for s in scans ]

    def prepare_desc(self,scan,stats):
        '''
        Convert individual scan to a set of words.
        '''
        s=scan
        text=[]
        for key, vals in s.items():
            if key in self._string_tags:
                try:
                    text+=[f"{key}_{val}" for val in vals.split(" ") if len(val)>0 ]
                except Exception as e:
                    print('WARNING: value error for key {}, value {}'.format(key,vals))
                    #raise ValueError("key: {}, value: {}".format(key,vals))
                    
            elif key in self._string_array_tags:
                try:
                    text+=[ f"{key}_{val}" for i in range(len(vals)) if len(vals[i])>0 ]
                except Exception as e:
                    print('WARNING: value error for key: {}, value: {}'.format(key,vals))
            elif key in self._num_tags:
                
                         
        return ' '.join([w for w in text ]) #if ((not w.isdigit()) and (len(w)>1))    
    
        