#!/bin/bash
echo '-----------------------------------------------------------'
echo 'Welcome to the get data script'
echo 'This script installs sshfs, mounts the linux cluster storage or the DSS storage and copies it to the server ssd'
echo '-----------------------------------------------------------'
echo 'Have you already started byobu? It is recommended to start byobu before continuing'
echo 'It is also helpful to open a new byobu window so that you can do other things while this script is running'
echo 'To open a new window, press F2, to change between existing windows, press F3'
echo '-----------------------------------------------------------'
while true; do
    read -p "Continue? (y/n)" yn
    case $yn in
        [Yy]* ) echo '-----------------------------------------------------------'
                read -p "Please enter your LRZ id: " LRZ_ID
                read -p "Which storage would you like to mount? 
                         Enter 1 for Tom's directory in the Linux cluster storage
                         Enter 2 for your personal DSS directory: " MOUNT_OPT
                echo '-----------------------------------------------------------'
                echo 'Installing sshfs';
		        sudo apt install sshfs; 
                y
                echo '-----------------------------------------------------------'
		        echo 'preparing new directories';
		        mkdir -p -v /ssdtemp/mounted; 
		        mkdir -p -v /ssdtemp/local;
                echo '-----------------------------------------------------------'
		        echo 'mounting storage via sshfs, you may be asked for your password';
                case $MOUNT_OPT in
                     [1] ) sshfs $LRZ_ID@lxlogin6.lrz.de:../di52tig /ssdtemp/mounted;
                     [2] ) sshfs $LRZ_ID@lxlogin6.lrz.de:/dss/dssfs01/pn69za/pn69za-dss-0001/$LRZ_ID /ssdtemp/mounted;
		        echo 'mounting completed';
                echo '-----------------------------------------------------------'
		        echo 'copying data from the mounted di-lab directory, this will take some time
                      you will find the local data in /ssdtemp/local/di-lab';
        		cp -r -v /ssdtemp/mounted/di-lab /ssdtemp/local;
		        echo 'data copied';
                echo '-----------------------------------------------------------'
                echo 'The get data script end here. Goodbye'
                echo '-----------------------------------------------------------'
		        break;;
		
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done 
