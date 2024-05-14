tag=0.2
sudo docker tag registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier:latest registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier:$tag
sudo docker login registry.nrg.wustl.edu
sudo docker push registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier:$tag

