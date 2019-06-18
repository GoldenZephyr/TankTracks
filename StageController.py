#!/usr/bin/python3
import signal
import time

import pyAPT
import zmq

import Parameters as p

keep_running = True
def sigint_handler(signo, stack_frame):
    print('Caught ctrl-c!')
    global keep_running
    keep_running = False

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    x_stage = pyAPT.LTS300(serial_number=p.LTS300_X)
    y_stage = pyAPT.LTS300(serial_number=p.LTS300_Y)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect('tcp://localhost:%d' % p.TRACK_PORT)
    topicfilter = ''
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    while keep_running:
        track_pos_str = socket.recv_string()
        track_toks = track_pos_str.split(' ')
        dx = float(track_toks[0])
        dy = float(track_toks[1])
        print('%f, %f' % (dx, dy))
       
        if abs(dx / p.X_MOVE_SCALE) > 0.25: # deadband check
            print('Issuing x stage command: %f' % (dx / p.X_MOVE_SCALE))
            x_stage.move(dx/p.X_MOVE_SCALE, wait=False)
        if abs(dy / p.Y_MOVE_SCALE) > 0.25: # deadband check
            print('Issuing y stage command: %f' % (-dy / p.Y_MOVE_SCALE))
            y_stage.move(-dy/p.Y_MOVE_SCALE, wait=False)
        time.sleep(3)
        print('finished loop')

    time.sleep(1)

    x_stage.close()
    y_stage.close()
#    x_stage.stop(wait=False)
#    y_stage.stop(wait=False)
#
#    x_stage._devce.close()
#    y_stage._devce.close()

if __name__ == '__main__':
    main()
