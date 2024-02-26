mkdir -p `pwd`/output
echo docker run --network="host" -t -i --rm -u $(id -u ${USER}):$(id -g ${USER}) -v `pwd`/output:/output -v `pwd`:/input registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier:latest run_classifier_xnat \"$@\"
docker run --network="host" -t -i --rm -u $(id -u ${USER}):$(id -g ${USER}) -v `pwd`/output:/output -v `pwd`:/input registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier:latest run_classifier_xnat "$@"
