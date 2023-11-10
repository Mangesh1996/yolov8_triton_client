import cv2
import numpy as np
from trion_utils import Detection_prediction
import torch
import os

def inference(path,labels_name,save):
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
                    if save:
                        save_path=os.path.join(os.getcwd(),"output")
                        os.makedirs(save_path,exist_ok=True)
                        cv2.imwrite(os.path.join(os.getcwd(),"output",filename),orig_img)
                    else:
                        # Display the image
                        cv2.imshow(filename, orig_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

def label_read():
    
    with open("/home/easemyai/Documents/Desktop_Items/nvidia_triton_server/trion_work/trion_server/triton_clients_onnx/labels.txt","r") as f:
        lines=f.readlines()
        print(lines)
        return [line[:-1].split(",")[0] for line in lines]
print(label_read())   
# if __name__=="__main__":
#     labels_name={0:"helment",1:"vest",2:"head",3:"person"} 
#     inference("/home/easemyai/Music/telegram_bot_data/data",labels_name,save=True)