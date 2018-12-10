#!/bin/bash
echo '-----------------------------------------------------------'
echo 'Welcome to the prepare_docker script!'
echo 'This script installs the requirements like numpy, and runs tests.'
echo '-----------------------------------------------------------'
echo 'install requirements'
echo '-----------------------------------------------------------'
conda install --file '/mnt/local/di-lab/test_scripts/requirements.txt'
y
echo '-----------------------------------------------------------'
echo 'run tests'
python3 /mnt/local/di-lab/test_scripts/test_modules.py
echo '-----------------------------------------------------------'
echo 'The prepare_docker script ends here. Goodbye'
echo '-----------------------------------------------------------'

