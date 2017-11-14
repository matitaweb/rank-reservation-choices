#! /bin/bash
sudo apt-get install ppa-purge
sudo ppa-purge ppa:mc3man/trusty-media 
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get install ffmpeg