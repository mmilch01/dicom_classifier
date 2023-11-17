#!/bin/bash

echo "building Docker image."
docker build . -t registry.nrg.wustl.edu/docker/nrg-repo/dicom_classifier
