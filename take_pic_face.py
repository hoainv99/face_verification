import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime
from config import get_config
from PIL import Image
import numpy as np
conf = get_config(False)
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
args = parser.parse_args()
from pathlib import Path
data_path = Path('data')
save_path_face = data_path/'face'

if not save_path_face.exists():
    save_path_face.mkdir()


# 初始化摄像头
cap = cv2.VideoCapture(0)
# 我的摄像头默认像素640*480，可以根据摄像头素质调整分辨率
cap.set(3,512)
cap.set(4,256)

mtcnn = MTCNN()

while cap.isOpened():
   
    isSuccess,frame = cap.read()

    if isSuccess:
        # frame_text = cv2.putText(frame,
        #             'Press t to take a picture,q to quit.....',
        #             (10,20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             1,
        #             (0,255,0),
        #             1,
        #             cv2.LINE_AA)
        cv2.imshow("Press t to take a picture,q to quit.....",frame)

    if cv2.waitKey(1)&0xFF == ord('t'):
        p =  Image.fromarray(frame[...,::-1])
        try:            
            bounding_boxes,_ = mtcnn.detect_faces(p)

            for box in bounding_boxes:
                x,y,x1,y1,s = box.astype(np.int32)
                image_face = frame[y:y1,x:x1,:]
            # print(image_face.shape)
            cv2.imwrite(str(save_path_face/'{}.jpg'.format(args.name)), image_face)
            print("get face succesfully")
        except:
            print('no face captured')
        
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cap.release()
cv2.destoryAllWindows()
