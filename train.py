import torch
from ultralytics import YOLO

def run ():
    torch.multiprocessing.freeze_support ()
    model=YOLO ("yolov8m-p2.yaml").load ("yolov8m.pt")
    model.train (
        data ='./datasets/data.yaml',
        epochs =100 , 
        optimizer ='Adam', 
        batch =16 , 
        imgsz =640 , 
        lr0 =0.0001 ,
        augment =True ,
        hsv_h =0.015 ,
        hsv_s =0.7 ,
        hsv_v =0.4 ,
        translate =0.2 ,
        scale =0.9 ,
        fliplr =0.5
    )

if __name__=='__main__':
    run ()