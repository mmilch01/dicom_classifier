tag=0.18
docker tag mmilch01/dicom_classifier:latest mmilch01/dicom_classifier:$tag
docker login docker.io
docker push mmilch01/dicom_classifier:$tag
