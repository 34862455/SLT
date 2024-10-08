import torch
import torch.nn as nn
from torchvision.io import read_image,read_video
import torch.nn.functional as F
from torchvision import transforms as t
import cv2
import mediapipe as mp
import json

import matplotlib.pyplot as plt
import copy

import numpy as np
import csv
import random
import os
import pickle
import gzip
import tqdm

from model_s3d import S3D
from PIL import Image, ImageDraw

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')

s3d = S3D(400)
num_classes = 2000
first = True
s3d.replace_logits(num_classes)
s3d.load_state_dict(torch.load('checkpoints/nslt_2000_066704_0.408206.pt'))


# Uncomment below if using SASL pretrained model
"""
weight_dict = torch.load('/home/botlhale/Documents/Mokgadi_masters/checkpoints/Sign2Text encoder/sign_sample_model/43200.ckpt')["model_state"]
model_dict = s3d.state_dict()
for name, param in weight_dict.items():
    if 's3d' in name:
        name = name.replace("s3d.","")
    
    if name in model_dict:
        if first:
            print("Yes")
            first = False
        if param.size() == model_dict[name].size():
            model_dict[name].copy_(param)
        else:
            print (' size? ' + name, param.size(), model_dict[name].size())
"""    
        

s3d.cuda()
for _, p in s3d.named_parameters():
    p.requires_grad = False

#Build Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
	# image.flags.writable = False				 # Image is no longer writable
	results = model.process(image)				 # Make prediction
	# image.flags.writable = True				 # Image is now writable
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
	return image, results
	
def draw_landmarks(image, results):
	mp_drawing.draw_landmarks(
	image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
	mp_drawing.draw_landmarks(
	image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
	mp_drawing.draw_landmarks(
	image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
	mp_drawing.draw_landmarks(
	image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
	
def draw_styled_landmarks(image, results):
	# Draw face connections
	mp_drawing.draw_landmarks(
	image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
	mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
	mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
	# Draw pose connections
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
							mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
							mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
							)
	# Draw left hand connections
	mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
							mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
							mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
							)
	# Draw right hand connections
	mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
							mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
							mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
							)


def make_dataset(feature_root, annotation_file):
    dataset = []
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                
                files = sorted([f for f in os.listdir(os.path.join(feature_root,row[0])) if f.endswith('.png')])
                text = row[6].lower()
                name = row[0]
                signer = row[4]
                gloss = row[5]
                dataset.append((files,name,signer,gloss,text))
                line_count += 1
    
    return dataset

def sort_key(x):
    return int((x.split('_')[1]).replace(".png",""))

def make_dataset_SASL(feature_root, annotation_file):
    folders = os.listdir(feature_root)

    annotation_file = list(filter(lambda x: x["file"].replace(".bag","") in folders, annotation_file ))
    # with open("/home/botlhale/Documents/Mokgadi_masters/SASL_new/SASL/vid_annotations.json","r") as file:
    #     annotations = json.load(file)
    dataset = []
    for row in annotation_file:
            files = sorted([f for f in os.listdir(os.path.join(feature_root,row["file"].replace(".bag",""))) if f.endswith('.png')])
            files.sort(key=sort_key)
            text = row["trans"]
            name = row["file"].replace(".bag","")
            signer = ""
            gloss = ""
            dataset.append((files,name,signer,gloss,text))
            
    
    return dataset

def pickle_features(feature_root, dataset,mode):
    data = []
    count = 0    
    for files,name,signer,gloss,text in dataset:  
        count+=1
        frames = []
        print(len(files), count)
        """ if len(files) > 210:
            files = files[:210] """
        for frame in files:
            frame_count = 0
            if int(name[4:6]) < 11 and frame_count%2!=0:
                frame_count+=1
            else:
                try:
                    img = read_image(os.path.join(feature_root,name,frame))
                    img = t.functional.resize(img,[200,200])
                    img = (img / 255.) * 2 - 1
                    frames.append(img)
                except:
                    print(os.path.join(feature_root,name,frame))
                frame_count+=1
        # for i in range(8-(len(frames)%8)):
        #     frames.append(torch.zeros(1,3,260,210))
        # frames = np.asarray(frames, dtype=np.float32)
        
        frames = torch.stack(frames,dim=0).float().to(device)
        
        #frames = frames.unfold(0,16,8)
        frames = frames.unsqueeze(0).permute(0,2,1,3,4)
        features = s3d(frames)
        print(features.squeeze(0).shape)
        data.append( {
            "name" : name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": features.squeeze(0)
        })
        
        
    with gzip.open('data/DSG_{}.pt'.format(mode), "wb") as f:
        pickle.dump(data, f)  

def pickle_features_mask(feature_root, dataset,mode):
    data = []
    count = 0    
    for files,name,signer,gloss,text in tqdm.tqdm(dataset):  
        count+=1
        frames = []
        # print(len(files), count)
        IMAGE_FILES = files

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for idx, file in enumerate(IMAGE_FILES):
                face_x = []
                face_y = []
                hands = []
                # print(os.path.join(feature_root,name,file))
                image = cv2.imread(os.path.join(feature_root,name,file))
                if image is not None:
                    image_height, image_width, _ = image.shape
                    image, results = mediapipe_detection(image, holistic) 
                    hands_in_face = False
                    if results.face_landmarks:
                        for keypoint in results.face_landmarks.landmark:
                            face_x.append(keypoint.x)
                            face_y.append(keypoint.y)
                            
                    
                        if results.left_hand_landmarks:
                            for keypoint in results.left_hand_landmarks.landmark:
                                hands.append((keypoint.x,keypoint.y))
                                
                                
                        
                        if results.right_hand_landmarks:
                            for keypoint in results.right_hand_landmarks.landmark:
                                hands.append((keypoint.x,keypoint.y))
                                
                        min_x_face = min(face_x)
                        max_x_face = max(face_x)
                        min_y_face = min(face_y)
                        max_y_face = max(face_y)
                        
                        for x, y in hands:
                            if x > min_x_face and x < max_x_face and y > min_y_face and y < max_y_face:
                                hands_in_face = True
                                break
                            
                    
                    # print(image_height,image_width)
                    newImg = Image.open(os.path.join(feature_root,name,file))
                    if not hands_in_face:
                        rect_coords = (min_x_face*image_width,min_y_face*image_height,max_x_face*image_width,max_y_face*image_height)
                        draw = ImageDraw.Draw(newImg)
                        draw.rectangle(rect_coords, fill="black")
                    newImg.save("tmpImg.png")
                
                    frame_count = 0
                    if int(name[4:6]) < 11 and frame_count%2!=0:
                        frame_count+=1
                    else:
                        try:
                            img = read_image("tmpImg.png")
                            # img = read_image(os.path.join(feature_root,name,file))
                            img = t.functional.resize(img,[200,200])
                            img = (img / 255.) * 2 - 1
                            frames.append(img)
                            
                        except:
                            print(os.path.join(feature_root,name,file))
                        frame_count+=1
            # for i in range(8-(len(frames)%8)):
            #     frames.append(torch.zeros(1,3,260,210))
            # frames = np.asarray(frames, dtype=np.float32)
            
            frames = torch.stack(frames,dim=0).float().to(device)
            
            #frames = frames.unfold(0,16,8)
            frames = frames.unsqueeze(0).permute(0,2,1,3,4)
            features = s3d.extract_features(frames)
            # print(features.squeeze(0).shape)
            data.append( {
                "name" : name,
                "signer": signer,
                "gloss": gloss,
                "text": text,
                "sign": features.squeeze(0)
            })
            
        
    with gzip.open('data/SASL_masked_{}.pt'.format(mode), "wb") as f:
        pickle.dump(data, f)  

def pickle_keypoint_features(feature_root, dataset,mode):
    data = []
    count = 0    
    for files,name,signer,gloss,text in dataset:  
        count+=1
        frames = []
        frames = []
        IMAGE_FILES = files

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for idx, file in enumerate(IMAGE_FILES):
                keypoints = []
                image = cv2.imread(os.path.join(feature_root,name,file))
                image_height, image_width, _ = image.shape

                
                image, results = mediapipe_detection(image, holistic)
                

                if results.pose_landmarks:
                    for keypoint in results.pose_landmarks.landmark[0:25]:
                        keypoints.append(keypoint.x)
                        keypoints.append(keypoint.y)
                        keypoints.append(keypoint.z)
                        keypoints.append(keypoint.visibility)
                else:
                    for i in range(25):
                        keypoints += [0,0,0,0]    

                if results.face_landmarks:
                    for keypoint in results.face_landmarks.landmark:
                        keypoints.append(keypoint.x)
                        keypoints.append(keypoint.y)
                        keypoints.append(keypoint.z)
                        keypoints.append(keypoint.visibility)
                else:
                    for i in range(468):
                        keypoints += [0,0,0,0]
                if results.left_hand_landmarks:
                    for keypoint in results.left_hand_landmarks.landmark:
                        keypoints.append(keypoint.x)
                        keypoints.append(keypoint.y)
                        keypoints.append(keypoint.z)
                        keypoints.append(keypoint.visibility)
                else:
                    for i in range(21):
                        keypoints += [0,0,0,0]
                if results.right_hand_landmarks:
                    for keypoint in results.right_hand_landmarks.landmark:
                        keypoints.append(keypoint.x)
                        keypoints.append(keypoint.y)
                        keypoints.append(keypoint.z)
                        keypoints.append(keypoint.visibility)

                else:
                    for i in range(21):
                        keypoints += [0,0,0,0]

                keypoints = torch.FloatTensor(keypoints)
        
                frames.append(keypoints.unsqueeze(0))
                
                
                
        frames = torch.cat(frames,dim=0)
        data.append( {
            "name" : name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": frames
        })
        print(count, frames.shape)
        if count%300 == 0 and count!=0:
            with gzip.open('data/DSG_keypoints_{0}_{1}.pt'.format(mode, count), "wb") as f:
                pickle.dump(data, f)
        
        
    with gzip.open('data/DSG_keypoints_{}.pt'.format(mode), "wb") as f:
        pickle.dump(data, f)

def make_dataset_vids(feature_root, annotation_file):
    dataset = []
    with open(annotation_file, encoding = 'cp850') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # if os.path.exists(feature_root+row[3]+'.mp4') and len(row[6].split()) < 20:
                if os.path.exists(feature_root+row[3]+'.mp4'):
                    text = row[6].lower()
                    name = row[3]
                    signer = ""
                    gloss = ""
                    dataset.append((name,signer,gloss,text))
    
    return dataset

def pickle_features_vids(feature_root, dataset,mode):
    data = []
    count = 0  
    with open('ASL_few_frames_{}.txt'.format(mode), 'w', encoding="utf8") as the_file:  
        for name,signer,gloss,text in dataset:  
            count+=1
            frames, _, _ = read_video(feature_root+name+'.mp4')
            # grey = t.Grayscale(3)
            frames = frames.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            # frames = grey(frames)
            frames = frames.permute(1, 0, 2, 3)  # (N, C, H, W,) -> (C, N, H, W)
            frames = t.functional.resize(frames,[200,200])
            
            if frames.size(dim=1) > 500:
                frames = frames[:,:500,:,:]
            elif frames.size(dim=1) < 16:
                the_file.write(('|'.join([name,text,str(frames.size(dim=0))])) + '\n')
                continue
            # elif frames.size(dim=0)%8 > 0:
            #     frames = torch.cat([frames,torch.zeros(8-(frames.size(dim=0)%8),3,200,200)], dim=0)
            # print(frames.size())
            frames = (frames / 255.)* 2 - 1
            print(frames.size(),count)
            #frames = frames.unfold(0,16,8)
            
            #frames = frames.permute(0,1,4,2,3)
            
            features = s3d.extract_features(frames.unsqueeze(0).to(device))
            #print(features.shape)
            print(features.squeeze(0).shape)
            data.append( {
                "name" : name,
                "signer": signer,
                "gloss": gloss,
                "text": text,
                "sign": features.squeeze(0)
            })
        
        #print(activation["avgpool"].squeeze().squeeze().shape)
        #dataset.append(item)
    with gzip.open('data/ASL_{}.pt'.format(mode), "wb") as f:
        pickle.dump(data, f)                 

"""
~~~~~~~~~~~~~~~~~ DSG features ~~~~~~~~~~~~~~~~~~~~
feature_root_train = '../train'
annotation_file_train = '../PHOENIX-2014-T.train.corpus.csv'
feature_root_val = '../dev'
annotation_file_val = '../PHOENIX-2014-T.dev.corpus.csv'
feature_root_test = '../dev'
annotation_file_test = '../PHOENIX-2014-T.dev.corpus.csv'
vocab_file = '../PHOENIX-2014-T.train.corpus.csv'


dataset = make_dataset(feature_root_train,annotation_file_train)
print(len(dataset))
pickle_features(feature_root_train,dataset,"train")
dataset = make_dataset(feature_root_test,annotation_file_test)
print(len(dataset))
pickle_features(feature_root_test,dataset,"test(dev)")
dataset = make_dataset(feature_root_val,annotation_file_val)
print(len(dataset))
pickle_features(feature_root_val,dataset,"dev")

"""

"""
~~~~~~~~~~~~~~~~~~~ SASL features ~~~~~~~~~~~~~~~~~~~~
feature_root =  "E:/Masters/SASL Corpus png cropped"

with open("E:\\Dataset final\\final_no_duplicates.json","r") as file:
    dataset = json.load(file)

cut_off = int(len(dataset)*0.06)

annotation_file_train = dataset[2*cut_off:]

annotation_file_val = dataset[0:cut_off]

annotation_file_test = dataset[cut_off:2*cut_off]


dataset = make_dataset_SASL(feature_root,annotation_file_train)
print(len(dataset))
pickle_features(feature_root,dataset,"train")
dataset = make_dataset_SASL(feature_root,annotation_file_test)
print(len(dataset))
pickle_features(feature_root,dataset,"test")
dataset = make_dataset_SASL(feature_root,annotation_file_val)
print(len(dataset))
pickle_features(feature_root,dataset,"dev")
"""


""" 
~~~~~~~~~~~~~~~~~~ How2Sign features ~~~~~~~~~~~~~~~~~~~~
feature_root_train = '/home/botlhale/Documents/Mokgadi_masters/How2Sign/train/raw_videos/'
annotation_file_train = '/home/botlhale/Documents/Mokgadi_masters/How2Sign/how2sign_realigned_train.csv'
feature_root_val = '/home/botlhale/Documents/Mokgadi_masters/How2Sign/val/raw_videos/'
annotation_file_val = '/home/botlhale/Documents/Mokgadi_masters/How2Sign/how2sign_realigned_val.csv'
feature_root_test = '/home/botlhale/Documents/Mokgadi_masters/How2Sign/test/raw_videos/'
annotation_file_test = '/home/botlhale/Documents/Mokgadi_masters/How2Sign/how2sign_realigned_test.csv'


dataset = make_dataset_vids(feature_root_train,annotation_file_train)
print(len(dataset))
pickle_features_vids(feature_root_train,dataset,"train")
dataset = make_dataset_vids(feature_root_test,annotation_file_test)
print(len(dataset))
pickle_features_vids(feature_root_test,dataset,"test")
dataset = make_dataset_vids(feature_root_val,annotation_file_val)
print(len(dataset))
pickle_features_vids(feature_root_val,dataset,"dev") """