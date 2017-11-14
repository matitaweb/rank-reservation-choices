#! /bin/bash


# install java8
# https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04
# http://blog.prabeeshk.com/blog/2016/12/07/install-apache-spark-2-on-ubuntu-16-dot-04-and-mac-os/
sudo apt-add-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo apt-get install git

sudo curl https://d3kbcqa49mib13.cloudfront.net/spark-2.2.0-bin-hadoop2.7.tgz | tar xvz -C .