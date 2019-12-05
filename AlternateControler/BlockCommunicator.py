import zmq
import params.Params as Params
import multiprocessing as mp
import os
import socket
import re
from AlternateControler.NetworkMessageType import NetworkMessageType
import time

def ZmqCom(comPipe):
    print("Com Process started : {}".format(os.getpid()))
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://{}:{}".format(Params.HOST, Params.PORT))
    print("Binded to tcp://{}:{}".format(Params.HOST, Params.PORT))
    while True:
        message = socket.recv()
        print("Received from client {}:{}".format(os.getpid(), message))

        '''try:
            pipeVal = comPipe.get_nowait()
            print("{}:{}".format(os.getpid(), pipeVal))
        except Exception as e:
            pass

        comPipe.put(10, False)'''

        if comPipe.poll():
            pipeVal = comPipe.recv()
            print("Received {}:{}".format(os.getpid(), pipeVal))

        socket.send(b"Tutu")

def TcpSocketCom(comPipe):
    inOut = "0,0"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((Params.HOST, Params.PORT))
        while True:
            print("Listening...")
            server.listen()
            conn, addr = server.accept()
            with conn:
                print("Connection from {}".format(addr))
                while True:
                    if comPipe.poll():
                        pipeVal = comPipe.recv()
                        inOut = pipeVal
                        print("Received from opencv {}:{}".format(os.getpid(), pipeVal))

                    data = conn.recv(1024)
                    if not data:
                        break
                    print("Received from client : {}".format(data))
                    strData = data.decode('utf-8')
                    reqCode, arg = ParseReceivedMsg(strData)
                    if reqCode == NetworkMessageType.ASK_SHAPE_INOUT.value[0]:  # why [0] ??
                        print("Ask In Out")
                        response = inOut
                    elif reqCode == NetworkMessageType.CHANGE_TEMPLATE.value[0]:  # why [0] ??
                        print("Ask Change template to {}".format(arg))
                        comPipe.send(arg)
                        baseTime = time.time()
                        while True:  # Avoid looking for useless inOut values
                            pipeVal = comPipe.recv()
                            if type(pipeVal) is int:  # better check ?
                                response = str(pipeVal)
                                if pipeVal == NetworkMessageType.TEMPLATE_CHANGED.value[0]:
                                    print('Template changed to {}'.format(arg))
                                elif pipeVal == NetworkMessageType.TEMPLATE_UNKNOWN.value[0]:
                                    print('Template {} unknown'.format(arg))
                                break
                            if time.time() - baseTime > 5:
                                print('Template change timed out.')
                                response = str(NetworkMessageType.ERROR_TEMPLATE_CHANGE.value[0])
                    else:  # Most likely impossible
                        print('Unknown request code {}'.format(reqCode))
                        response = 'BUG'

                    conn.send(response.encode())

def ParseReceivedMsg(msg):
    regex = r'^\[(\d+)\]-(\d*)$'
    match = re.search(regex, msg)
    if match:
        arg = int(match.group(2)) if len(match.group(2)) > 0 else ''
        return int(match.group(1)), arg
    else:
        print('Invalid network message format')
        return 0, 0


def BlockCommunicator(comPipe):
    TcpSocketCom(comPipe)

