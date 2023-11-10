import triton
import cv2
import torch
from copy import deepcopy
import time
import numpy as np
import torchvision


def preprocess(img):
    """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
    not_tensor=not isinstance(img,torch.Tensor)
    if not_tensor:
        
        im=np.stack(np.expand_dims(img,axis=0))
        
        im=im[...,::-1].transpose((0,3,1,2))# BGR to RGB,BHWC to BCHW,(n,3,h,w)
        
        im=np.ascontiguousarray(im) #contiguous
        im=torch.from_numpy(im)
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    im=im.to(device)
    im=im.float()
    
    if not_tensor:
        im /= 255
    
    return im
def inferece(image,url):
    model=triton.TritonRemoteModel(url=url)
    img=preprocess(image).cpu().numpy()
    y_pred=model(img)
    if isinstance(y_pred, (list, tuple)):
        return torch.from_numpy(y_pred[0]) if len(y_pred) == 1 else [torch.from_numpy(x) for x in y_pred],preprocess(image)
    else:
        return torch.from_numpy(y_pred),preprocess(image)
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y
def non_max_suppression(prediction,conf_thres=0.25,iou_thres=0.45,classes=None,labels=(),max_det=300,nc=0,max_time_img=0.05,max_nms=30000,max_wh=7680):
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction,(list,tuple)):
        prediction=prediction[0]
    device=prediction.device
    mps="mps" in device.type #Apple mps
    if mps: # # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction=prediction.cpu()
    bs=prediction.shape[0] #batch size
    nc=nc or (prediction.shape[1]-4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc #mask  start index
    xc=prediction[:,4:mi].amax(1) > conf_thres #candidates

    #setting
    time_limit=0.5 + max_time_img * bs # second to quit after
    prediction=prediction.transpose(-1,-2)  #shape (1,84,6300) to shape(1,6300,84)
    prediction[...,:4] = xywh2xyxy(prediction[...,:4]) #xywh to xyxy
    t=time.time()
    output=[torch.zeros((0,6+nm),device=prediction.device)] *bs
    for xi,x in enumerate(prediction): #image index,image inference
        x=x[xc[xi]] #confidence
        if not x.shape[0]:
            continue
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        conf,j=cls.max(1,keepdim=True)
        x=torch.cat((box,conf,j.float(),mask),1)[conf.view(-1)>conf_thres]
        n=x.shape[0] # number of box
        if not n: #no box
            continue
        if n>max_nms: # excess boxes
            x=x[x[:,4].argsort(descending=True)[:max_nms]] # sort by confidence and remove excess boxes
        #Batched NMS
        c=x[:,5:6]*(max_wh) # classes
        boxes,scores=x[:,:4] +c,x[:,4] #boxes (offset by class) ,scores
        i=torchvision.ops.nms(boxes,scores,iou_thres)
        i=i[:max_det] #limit detections
        output[xi]=x[i]

        if mps:
            output[xi]=output[xi].to(device)
    return output
def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    
    return boxes

def Results(names,boxes,img_shape):
    cls=boxes[:,-1] #class idxs
    cls=torch.Tensor.int(cls)
    conf=boxes[:,-2] #confidence
    xyxy=boxes[:,:4]
    meta_data={"class_index":cls,"confidence":conf,"bboxs":xyxy}
    return (meta_data)

def Detection_prediction(orig_image):
    predi,img0=inferece(orig_image,url=f'http://localhost:8000/yolov8')
    pre=non_max_suppression(prediction=predi)
    results=[]
    labels_name={0:"helment",1:"vest",2:"head",3:"person"} #class names
    for pred in (pre):
        pred[:,:4] =scale_boxes(img0.shape[2:],pred[:,:4],orig_image.shape)
        results.append(Results(names=labels_name,boxes=pred,img_shape=orig_image.shape))
    return (results)

