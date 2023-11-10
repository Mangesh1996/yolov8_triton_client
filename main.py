import cv2
import numpy as np
from trion_utils import Detection_prediction
import os
import gdown
import zipfile
def download_model():
    if not os.path.exists("model_respository"):
        print('Creating model repository')
        os.makedirs("model_repository")
        url="https://drive.usercontent.google.com/download?id=1ERMb5zpM620C_cB9HvtVduUtN-kAdFlQ&export=download&authuser=0&confirm=t&uuid=1a066ffe-6192-4919-adbe-6252497c2d73&at=APZUnTXixonXW1DWI3nlT-CP9wpN:1699619836338"
        output="model_repository/safety_kit.zip"
        gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(output,'r') as zip_ref:
            zip_ref.extractall("model_repository")
        os.remove(output)
        print('Model downloaded successfully!')
    else:
        print('Model already exists in the repository')
    
def inference(path,labels_path,save):
    download_model()
    labels_name=label_read(labels_path)
    for file_index,filename in enumerate(os.listdir(path)):
        if filename.endswith('.jpg'):            
            image_path=path+"/"+filename
            # Load your image
            orig_img = cv2.imread(image_path)
            orig_shape=orig_img.shape
            image=cv2.resize(orig_img,(640,640))
            resize_shape=image.shape
            meta_data=Detection_prediction(image)
            width_scale = orig_shape[1] / resize_shape[1]
            height_scale = orig_shape[0] / resize_shape[0]
            # Loop through the detections and draw bounding boxes
            for i in range(len(meta_data[0]["class_index"])):
                confidence = np.array(meta_data[0]["confidence"][i]).tolist()
                if confidence >=0.5:
                    class_index = int(meta_data[0]["class_index"][i])
                    box = meta_data[0]["bboxs"][i]
                    name = labels_name[class_index]
                    confidence=round(confidence, 2)
                    label = (f'{name} {confidence}' if confidence else name)
                    # Draw bounding box
                    box[0]=box[0]*width_scale
                    box[1]=box[1]*height_scale
                    box[2]=box[2]*width_scale
                    box[3]=box[3]*height_scale
                    color = (0, 255, 0) # Green color for the bounding box
                    cv2.rectangle(orig_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # Display confidence and class label
                
                    # Modify the parameters to put the label inside the detection box
                    cv2.putText(orig_img, label, (int(box[0] ), int(box[1] + label_size[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    continue
                if save:
                    save_path=os.path.join(os.getcwd(),"output")
                    os.makedirs(save_path,exist_ok=True)
                    cv2.imwrite(os.path.join(os.getcwd(),"output",filename),orig_img)
                else:
                    # Display the image
                    cv2.imshow(filename, orig_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

def label_read(labels_path):    
    with open(labels_path,"r") as f:
        labels=f.readlines()
        labels= {i:label.strip() for i,label in enumerate(labels)}
        return labels 
if __name__=="__main__":
    labels_path=os.path.join(os.getcwd(),"labels.txt")
    inference("images",labels_path,save=False)
    