#!/bin/bash

# Activate the micromamba base environment for every container start.
if [ -f "/opt/micromamba/etc/profile.d/micromamba.sh" ]; then
    source /opt/micromamba/etc/profile.d/micromamba.sh
    micromamba activate base
fi
