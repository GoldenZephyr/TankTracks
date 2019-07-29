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
    socket.setsockopt(zmq.RCVTIMEO, 250)
    socket.connect('tcp://%s:%d' % (p.TRACK_IP, p.TRACK_PORT))
    topicfilter = ''
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    pos_pub = context.socket(zmq.PUB)
    pos_pub.bind('tcp://*:%s' % p.STAGE_POSITION_PORT)

    x_stage = LTS300(p.LTS300_X)
    y_stage = LTS300(p.LTS300_Y)
    z_stage = LTS300(p.LTS300_Z)
    x_stage.initialize()
    y_stage.initialize()
    z_stage.initialize()

    time.sleep(2)
    x_stage.device.StartPolling(50)
    y_stage.device.StartPolling(50)
    z_stage.device.StartPolling(50)
    signal.signal(signal.SIGINT, sigint_handler)
    
    while keep_going:
        px = x_stage.device.Position
        py = y_stage.device.Position
        pz = z_stage.device.Position
        pos_pub.send_string('%s %s %s' % (px, py, pz))

        try:
            track_pos_str = socket.recv_string()
        except zmq.Again as e:
            print('No Message!')
            continue
        track_toks = track_pos_str.split(' ')
        dx = float(track_toks[0])
        dy = float(track_toks[1])
        dz = float(track_toks[2])
        print('%f, %f, %f' % (dx, dy, dz))

        # NOTE: this needs to be tuned/figured out. Here we need to map from pixel space to mm.
        # Currently it's just a random guess
        if abs(dx/p.X_MOVE_SCALE) > p.STAGE_DEADBAND:
            try:
                x_stage.jog(-dx/p.X_MOVE_SCALE, 0, 300)  # <- tune this (MOVE_SCALE)
            except:
                pass
        if abs(dy/p.Y_MOVE_SCALE) > p.STAGE_DEADBAND:
            try:
                y_stage.jog(-dy/p.Y_MOVE_SCALE, 0, 300)  # <- tune this (MOVE_SCALE)
            except:
                pass

        try:
            z_stage.jog(dz/p.Z_MOVE_SCALE, 20, 295)
            print('z_stage jog  set to %f' % (dz / p.Z_MOVE_SCALE))
        except:
            pass
            
        #time.sleep(.1)
        
    x_stage.device.StopImmediate()
    y_stage.device.StopImmediate()


if __name__ == '__main__':
    main()
