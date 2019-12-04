import zmq
import params.Params as Params
import multiprocessing as mp
import os
import socket

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((Params.HOST, Params.PORT))
        while True:
            print("Listening...")
            server.listen()
            conn, addr = server.accept()
            with conn:
                print("Connection from {}".format(addr))
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    print(data)
                    conn.send(b"Hello")


def BlockCommunicator(comPipe):
    TcpSocketCom(comPipe)

