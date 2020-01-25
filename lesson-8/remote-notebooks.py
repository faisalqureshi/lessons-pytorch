#!/usr/bin/env python
# coding: utf-8

# # Remote Jupyter Notebooks
# 
# Please see [https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh)

# On the remote machine, start the IPython notebooks server:
# 
# ~~~bash
# remote_user@remote_host$ ipython notebook --no-browser --port=8889
# ~~~
# 
# Usually IPython opens a browser to display the available notebooks, but we do not need that so we use the option --no-browser. We also change the port to 8889, for no other reason than to show how this is done.
# 
# On the local machine, start an SSH tunnel:
# 
# ~~~bash
# local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
# ~~~
# 
# The first option -N tells SSH that no remote commands will be executed, and is useful for port forwarding. The second option -f has the effect that SSH will go to background, so the local tunnel-enabling terminal remains usable. The last option -L lists the port forwarding configuration (remote port 8889 to local port 8888).
# 
# Now open your browser on the local machine and type in the address bar
# 
# ~~~
# localhost:8888
# ~~~
# 
# which displays your remotely running IPython notebook server.
# 
# To close the SSH tunnel on the local machine, look for the process and kill it manually:
# 
# ~~~bash
# local_user@local_host$ ps aux | grep localhost:8889
# local_user 18418  0.0  0.0  41488   684 ?        Ss   17:27   0:00 ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
# local_user 18424  0.0  0.0  11572   932 pts/6    S+   17:27   0:00 grep localhost:8889
# 
# local_user@local_host$ kill -15 18418
# ~~~
