import zmq
import Parameters as p
from Messages import AutofocusMessage, SetFocusMessage


def main():
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://*:%s' % p.AUTOFOCUS_PORT)

    socket_sub = context.socket(zmq.SUB)
    socket_sub.setsockopt(zmq.CONFLATE, 1)
    socket_sub.connect('tcp://%s:%d' % (p.CURRENT_FOCUS_IP, p.CURRENT_FOCUS_PORT))
    topicfilter = ''
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    while 1:
        inp = input('\nFull Autofocus: Enter\nCustom Autofocus: a enter\nRead Autofocus: b enter\nSet Focus: c enter\nRefine Focus: d enter\n>')
        if inp == '':
            af_msg = AutofocusMessage(0, 255, 10)
            socket_pub.send_pyobj(af_msg)
        elif inp == 'a':
            inp = input('Please enter custom autofocus: <min>.<max>.<step>: ')
            try:
                af_msg = AutofocusMessage(*[int(i) for i in inp.split('.')])
            except:
                print('Malformed command!')
                continue
            print([int(i) for i in inp.split('.')])
            socket_pub.send_pyobj(af_msg)
        elif inp == 'b':
            cfocus = socket_sub.recv()
            print(int(cfocus))
        elif inp == 'c':
            inp = input('Enter desired focus: ')
            try:
                focus = int(inp)
            except:
                print('Must enter valid integer for focus!')
                continue
            socket_pub.send_pyobj(SetFocusMessage(focus))
        elif inp == 'd':
            cfocus = int(socket_sub.recv())
            socket_pub.send_pyobj(AutofocusMessage(max(0, cfocus - 10), min(255, cfocus + 10), 1))
        else:
            print('Please enter a valid command')


if __name__ == '__main__':
    main()