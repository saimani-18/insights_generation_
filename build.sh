#!/bin/bash

apt-get update
apt-get install -y python3-distutils


pip install --disable-pip-version-check --target . --upgrade -r requirements.txt

