import multiprocessing as mp
from AlternateControler.BlockCommunicator import BlockCommunicator
from AlternateControler.BlockDetector import BlockDetector

if __name__ == '__main__':
    opencvPipe, comPipe = mp.Pipe()
    #comQ = mp.Queue()
    #comProcess = mp.Process(target=BlockCommunicator, args=(comPipe,))
    #comProcess.start()
    bDetector = BlockDetector(opencvPipe)
    bDetector.RunDetection()
    #comProcess.join(timeout=5)
