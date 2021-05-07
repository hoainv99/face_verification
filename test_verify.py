import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='verification')
    parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
    parser.add_argument('--threshold','-t', default=1.4, type=float,help='threshold of embedding')
    args = parser.parse_args()

    data_path = Path('data')
    save_path_card = data_path/'card'/f"{args.name}.jpg"
    save_path_face = data_path/'face'/f"{args.name}.jpg"    

    conf = get_config(False)
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    card = cv2.imread(str(save_path_card))
    face = cv2.imread(str(save_path_face))
    card = cv2.resize(card,(112,112))
    face = cv2.resize(face,(112,112))
    card_pil = Image.fromarray(card[:,:,::-1])
    face_pil = Image.fromarray(face[:,:,::-1])
    s_t = time.time()
    results, score = learner.infer(conf, face_pil, card_pil, False)
    print(time.time()-s_t)
    print(score)
    if score[0] < args.threshold:
        print("True")
    else:
        print("False")



     