tag=latest
docker tag registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier:latest mmilch01/dicom_classifier:$tag
docker login docker.io
docker push mmilch01/dicom_classifier:$tag
