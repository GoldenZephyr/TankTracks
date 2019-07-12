TankTracks Overview
===================

Tank tracks is a set of Python programs that enables the tracking of animals in
 tanks. A wide-view "spotter" camera tracks the bulk movement of the animal and
 a narrow field-of-view macro lens watches a feature of interest on the animal.
 For the sake of robustness, extensibility, and easy multiprocessing in light 
 of Python's GIL, there are several separate processes that communicate over
 ZMQ. The programs can be launched in any order, but certain groups of
 programs must be running in order to *do* anything. It is also possible to run
 the various modules on separate computers, as the processes communicate over
 TCP (however, note that apparently Windows is smart enough to bypass its actual
 TCP stack when communicating to the same machine). The correct IP for each
 module would have to updated in `Parameters.py`.

Main Program
============

`CameraInterface.py`
------------------
Module for interfacing with our two cameras (Point Gray and E-con Systems liquid lens camera).
This code grabs frames and publishes them. This functionality is broken out by itself
so that frames can be grabbed as quickly as possible, decoupling frame grabbing io from
future processing. Also the interface for focusing the liquid lens camera.

__Publishes__
* CURRENT_FOCUS
* VIDEO_{ZOOM | SPOTTER}

__Subscribes__
* AUTOFOCUS

__NOTES__
* should probably implement a zoom interface for the LL camera too

`CameraDisplaySpotter.py`
-----------------------
Module for displaying the "spotter camera". It also serves as the interface for tracking,
as the user must select what to track on the screen.

__Publishes__
* TRACK

__Subscribes__
* VIDEO_SPOTTER

`CameraDisplayZoom.py`
---------------------
NOT YET IMPLEMENTED. But  this will be a display of the macro lens camera. Similar to above module,
but probably without the same kind of tracking interface.

__Publishes__

__Subscribes__

`CameraSaver.py`
--------------
Module for saving video from either camera. This is separated from grabbing the frames
so that grabbing and saving can effectively be "pipelined." Also makes it easy to
start/stop saving video without disrupting other things that are running.

__Publishes__
* nothing

__Subscribes__
* VIDEO_{ZOOM | SPOTTER}

`StageControllerWin.py`
----------------------
Module for controlling the Thorlabs stages. Note that it would be relatively
simple to swap out the Thorlabs stages for other motors, this module
would just have to be rewritten.

__Publishes__
* CURRENT_POS (NB: Not Yet Implemented)

__Subscribes__
* TRACK

`Parameters.py`
---------------
This file stores all of the important parameters for TankTracks, including the
IP address and port number for each of the Pub/Sub topics, tracking parameters,
video settings, and more.


Utilities
=========

autofocus_sender.py
-------------------

__Publishes__
* AUTOFOCUS

__Subscribes__
* CURRENT_FOCUS
