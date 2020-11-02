#!/bin/bash
# open mynteye camera: modified file path plz
cd /home/nvidia/MYNT-EYE-S-SDK-master/wrappers/ros/
source ./devel/setup.bash
{
  gnome-terminal -x bash -c "roslaunch mynt_eye_ros_wrapper mynteye.launch"
}&

sleep 10s
# launch covins agent.launch: modified file path plz
cd /home/nvidia/covins/
source ./devel/setup.bash
{
  gnome-terminal -x bash -c "roslaunch vins_estimator agent.launch;exec bash"
}&