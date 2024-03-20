#!/bin/bash

echo "REMEMBER, MUST RUN AS LOCAL USER"

git pull

pushd juxnat_lib
        echo git pull
        git pull
popd

mkdir -p src
cp juxnat_lib/xnat_utils.py src/
cp *.py src/

echo chmod -R o+rX `pwd`
chmod -R o+rX `pwd`

