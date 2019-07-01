import time
from lts300 import LTS300
import zmq
import signal
import Parameters as p

keep_going = True


def sigint_handler(signo, stack_frame):
    global keep_going
    keep_going = False


def main():

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    socket.connect('tcp://%s:%d' % (p.TRACK_IP, p.TRACK_PORT))
    topicfilter = ''
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    x_stage = LTS300(p.LTS300_X)
    y_stage = LTS300(p.LTS300_Y)
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

        # NOTE: this needs to be tuned/figured out. Here we need to map from pixel space to mm.
        # Currently it's just a random guess
        if abs(dx/p.X_MOVE_SCALE) > p.STAGE_DEADBAND:
            x_stage.jog(-dx/p.X_MOVE_SCALE)  # <- tune this
        if abs(dy/p.Y_MOVE_SCALE) > p.STAGE_DEADBAND:
            y_stage.jog(dy/p.Y_MOVE_SCALE)  # <- tune this
            
        time.sleep(.1)
        
    x_stage.device.StopImmediate()
    y_stage.device.StopImmediate()


if __name__ == '__main__':
    main()
