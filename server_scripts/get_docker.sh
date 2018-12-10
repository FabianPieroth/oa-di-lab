#!/bin/bash
echo '-----------------------------------------------------------'
echo 'Welcome to the get docker script!
This script logs into the nvidia docker cloud, pulls the docker image and runs it. 
In the end, you enter the docker container and the script will end automatically.'
echo '-----------------------------------------------------------'
echo 'Have you already started byobu? It is recommended to start byobu before continuing'
echo 'It is also helpful to open a new byobu window so that you can do other things while this script is running'
echo 'To open a new window, press F2, to change between existing windows, press F3'
echo '-----------------------------------------------------------'
while true; do
    read -p "Continue? (y/n)" yn
    case $yn in
        [Yy]* ) echo '-----------------------------------------------------------'
                echo 'Logging into the Nvidia docker cloud'
                docker login nvcr.io;
                echo 'logged in';
                echo '-----------------------------------------------------------'
                echo 'pulling docker';
                docker pull nvcr.io/nvidia/pytorch:18.08-py3;
                echo '-----------------------------------------------------------'
                echo 'running docker'
                nvidia-docker run -it --rm -v /ssdtemp/:/mnt nvcr.io/nvidia/pytorch:18.08-py3;
                echo '-----------------------------------------------------------'
		        break;;
		
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done 

