source ~/ros_ws/devel/setup.bash
~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/run_sim.sh -vpg
gnome-terminal --tab -x bash -c "<cd ~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts; ./score_tracker.py>; exec bash"
gnome-terminal --tab -x bash -c "<>"

~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/score_tracker.py
python3 ~/ros_ws/src/controller_package/nodes/main.py