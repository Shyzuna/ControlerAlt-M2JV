import params.Params as Params
import multiprocessing as mp
import os
import socket
import re
from AlternateControler.NetworkMessageType import NetworkMessageType
import time

def TcpSocketCom(comPipe):
    bDetectVal = None
    newVal = False

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
                        bDetectVal = pipeVal # now receive both template in out & countdown stop/start
                        newVal = True
                        print("Received from opencv {}:{}".format(os.getpid(), pipeVal))

                    data = conn.recv(1024)
                    if not data:
                        break
                    #print("Received from client : {}".format(data))
                    strData = data.decode('utf-8')
                    reqCode, arg = ParseReceivedMsg(strData)
                    # Should split ask in out and countdown start/stop
                    if reqCode == NetworkMessageType.ASK_SHAPE_INOUT.value:
                        #print("Ask In Out")
                        if newVal:
                            response = bDetectVal
                            newVal = False
                        else:
                            response = 'Empty'
                    elif reqCode == NetworkMessageType.CHANGE_TEMPLATE.value:
                        print("Ask Change template to {}".format(arg))
                        comPipe.send(strData)
                        baseTime = time.time()
                        while True:  # Avoid looking for useless inOut values
                            pipeVal = comPipe.recv()
                            if type(pipeVal) is int:  # better check ?
                                response = str(pipeVal)
                                if pipeVal == NetworkMessageType.TEMPLATE_CHANGED.value:
                                    print('Template changed to {}'.format(arg))
                                elif pipeVal == NetworkMessageType.TEMPLATE_UNKNOWN.value:
                                    print('Template {} unknown'.format(arg))
                                break
                            if time.time() - baseTime > 5:
                                print('Template change timed out.')
                                response = str(NetworkMessageType.ERROR_TEMPLATE_CHANGE.value)
                                break
                    elif reqCode == NetworkMessageType.MENU_MODE.value:
                        print("Ask Menu mode")
                        comPipe.send(strData)
                        baseTime = time.time()
                        while True:  # Avoid looking for useless inOut values
                            pipeVal = comPipe.recv()
                            if type(pipeVal) is int:  # better check ?
                                response = str(pipeVal)
                                if pipeVal == NetworkMessageType.MENU_MODE_OP.value:
                                    print('Menu mode Op')
                                break
                            if time.time() - baseTime > 5:
                                print('Menu mode change timed out.')
                                response = str(NetworkMessageType.MENU_MODE_FAIL.value)
                                break
                    elif reqCode == NetworkMessageType.MENU_CHECK.value:
                        #print("Menu Check")
                        # Should split ask in out and countdown start/stop
                        if newVal:
                            response = bDetectVal
                            newVal = False
                        else:
                            response = 'Empty'
                    else:  # Most likely impossible
                        print('Unknown request code {}'.format(reqCode))
                        response = 'BUG'

                    conn.send(response.encode())
            print('Client disconnected')

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

