# dicom_classifier
DICOM classifier code.

usage: 
`run_classifier_xnat <input_dir> <configuration_file> [options]`

Runs in offline mode unless  XNAT credentials and required XNAT fields are specified.  
In online mode, sets the _scan type_ XNAT field to the inferred type.  
The input configuration file must point to correct model, nomenclature and tokenizer files.  

Can run on a single or multiple experiment directory (directory structure conforms to container service project or experiment mounts)  
single experiment directory: `<input_dir>/SCANS/{scan ID}/DICOM/{DICOM files}`  
multi-experiment directory: `<input_dir>/{experiment ID}/SCANS/{scan ID}/DICOM/{DICOM files}`  
required Python modules (autoencoder_models.py and dependencies) must be accessible under  directory.
  
options:  
```
  -output     <directory>         output directory [/output]  
  -jsession   <XNAT JSESSION>  Either this or user/password required in online mode  
  -server     <XNAT server>       Required in online mode  
  -project    <XNAT project>      Required in online mode 
  -user       <XNAT user>         This or JSESSION is required in online mode  
  -pass       <XNAT password>     This or JSESSION is required in online mode  
  -subject    <XNAT subject>      Required for single-experiment mode [online mode]  
  -experiment <XNAT experiment>   Required for single-experiment mode [online mode]  
```
