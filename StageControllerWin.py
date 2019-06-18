import zmq
import time
import sys
from lts300 import LTS300
import zmq
import signal



LTS300_X = '45874644'
LTS300_Y = '45874637'

keep_going = True

def sigint_handler(signo, stack_frame):
    global keep_going
    keep_going = False

def main():

    ip_addr = 'localhost'
    port_num = 5557

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    socket.connect('tcp://%s:%d' % (ip_addr, port_num))
    topicfilter = ''
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    x_stage = LTS300(LTS300_X)
    y_stage = LTS300(LTS300_Y)
    x_stage.initialize()
    y_stage.initialize()

    time.sleep(2)
    x_stage.device.StartPolling(50)
    y_stage.device.StartPolling(50)
    signal.signal(signal.SIGINT, sigint_handler)
    
    while keep_going:

        try:
            track_pos_str = socket.recv_string()
        except zmq.Again as e:
            print('Timed out!')
            time.sleep(1)
            continue
        track_toks = track_pos_str.split(' ')
        dx = float(track_toks[0])
        dy = float(track_toks[1])
        print('%f, %f' % (dx, dy))
    
        if abs(dx) > 30:
            x_stage.jog(-dx/20)
        if abs(dy) > 30:
            y_stage.jog(dy/20)
            
        time.sleep(.1)
        
    x_stage.device.StopImmediate()
    y_stage.device.StopImmediate()

if __name__== '__main__':
    main()