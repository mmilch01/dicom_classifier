#!/bin/bash

echo "building Docker image for the BOLD Extracted Networks Inferred by Classifier Extrapolation (BENICE) pipeline."
docker build . -t nrg/benice_docker
