import zmq
import signal
import time
import os

import win32file

import Parameters as p

keep_running = True


def sigint_handler(signo, stack_frame):
    global keep_running
    keep_running = False


def main():
    signal.signal(signal.SIGINT, sigint_handler)

    context = zmq.context()

    position_sub = context.socket(zmq.SUB)
    position_sub.setsockopt(zmq.RCVTIMEO, 0)
    position_sub.connect('tcp://%s:%d' % (p.STAGE_POSITION_IP, p.STAGE_POSITION_PORT))
    position_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    ll_sub = context.socket(zmq.SUB)
    ll_sub.setsockopt(zmq.RCVTIMEO, 0)
    ll_sub.connect('tcp://%s:%d' % (p.CURRENT_FOCUS_IP, p.CURRENT_FOCUS_PORT))
    ll_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    sharpness_macro_sub = context.socket(zmq.SUB)
    sharpness_macro_sub.setsockopt(zmq.RCVTIMEO, 0)
    sharpness_macro_sub.connect('tcp://%s:%d' % (p.MACRO_SHARPNESS_IP, p.MACRO_SHARPNESS_PORT))
    sharpness_macro_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    track_estimate_sub = context.socket(zmq.SUB)
    track_estimate_sub.setsockopt(zmq.RCVTIMEO, 0)
    track_estimate_sub.connect('tcp://%s:%d' % (None, None))
    track_estimate_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    track_deltas_sub = context.socket(zmq.SUB)
    track_deltas_sub.setsockopt(zmq.RCVTIMEO, 0)
    track_deltas_sub.connect('tcp://%s:%d' % (p.TRACK_IP, p.TRACK_PORT))
    track_deltas_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    wopen = lambda fn: win32file.CreateFile(fn, win32file.GENERIC_WRITE, win32file.FILE_SHARE_READ, None, win32file.OPEN_ALWAYS, win32file.FILE_ATTRIBUTE_NORMAL, None)

    fo_stage_position = wopen(os.path.join('logs', 'stage_position', 'current.csv'))
    fo_ll_setting = wopen(os.path.join('logs', 'll_setting', 'current.csv'))
    fo_sharpness_macro = wopen(os.path.join('logs', 'sharpness_macro', 'current.csv'))
    fo_track_estimate = wopen(os.path.join('logs', 'track_estimate', 'current.csv'))
    fo_track_deltas = wopen(os.path.join('logs', 'track_deltas', 'current.csv'))

    #fo_stage_position.write('time,x,y,z')
    #fo_ll_setting.write('time,ll_setting')
    #fo_sharpness_macro.write('time,macro_sharpness')
    #fo_track_estimate.write('time,track_x,track_y')
    #fo_track_deltas.write('time,dx,dy,dz')

    win32file.WriteFile(fo_stage_position, 'time,x,y,z')
    win32file.WriteFile(fo_ll_setting, 'time,ll_setting')
    win32file.WriteFile(fo_sharpness_macro, 'time,macro_sharpness')
    win32file.WriteFile(fo_track_estimate, 'time,track_x,track_y')
    win32file.WriteFile(fo_track_deltas, 'time,dx,dy,dz')

    while keep_running:

        try:
            pos_msg = position_sub.recv_string()
            nums = [float(x) for x in pos_msg.split(' ')]
            data_line = '%f,%f,%f,%f\n' % (time.time(), nums[0], nums[1], nums[2])
            #fo_stage_position.write(data_line)
            win32file.WriteFile(fo_stage_position, data_line)
        except zmq.Again:
            pass
        except Exception as ex:
            print(ex)
            print('Could not parse Position message: %s' % pos_msg)

        try:
            ll_msg = ll_sub.recv_string()
            num = float(ll_msg)
            data_line = '%f,%f\n' % (time.time(), num)
            #fo_ll_setting.write(data_line)
            win32file.WriteFile(fo_ll_setting, data_line)
        except zmq.Again:
            pass
        except Exception as ex:
            print(ex)
            print('Could not parse LL focus message: %s' % ll_msg)

        try:
            sharpness_macro_msg = sharpness_macro_sub.recv_string()
            num = float(sharpness_macro_msg)
            data_line = '%f,%f\n' % (time.time(), num)
            #fo_sharpness_macro.write(data_line)
            win32file.WriteFile(fo_sharpness_macro, data_line)
        except zmq.Again:
            pass
        except Exception as ex:
            print(ex)
            print('Could not parse macro_sharpness message: %s' % sharpness_macro_msg)

        try:
            track_estimate_msg = track_estimate_sub.recv_string()
            num = [float(x) for x in track_estimate_msg.split(' ')]
            data_line = '%f,%f,%f\n' % (time.time(), num[0], num[1])
            #fo_track_estimate.write(data_line)
            win32file.WriteFile(fo_track_estimate, data_line)
        except zmq.Again:
            pass
        except Exception as ex:
            print(ex)
            print('Could not track estimate message: %s' % track_estimate_msg)

        try:
            track_deltas_msg = track_deltas_sub.recv_string()
            num = [float(x) for x in track_deltas_msg.split(' ')]
            data_line = '%f,%f,%f,%f\n' % (time.time(), num[0], num[1], num[2])
            #fo_track_deltas.write(data_line)
            win32file.WriteFile(fo_track_deltas, data_line)
        except zmq.Again:
            pass
        except Exception as ex:
            print(ex)
            print('Could not track estimate message: %s' % track_deltas_msg)

    fo_stage_position.close()
    fo_ll_setting.close()
    fo_sharpness_macro.close()
    fo_track_estimate.close()
    fo_track_deltas.close()


if __name__ == '__main__':
    main()
