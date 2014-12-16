#!/bin/bash
cp network_my.cc ./New_backup/
cp network_backup.cc ./New_backup/
cp network.cc ./New_backup
 
mv network.cc network_my.cc
mv network_backup.cc network.cc
