#!/bin/bash

echo "building Docker image."
docker build . -t mmilch01/dicom_classifier
