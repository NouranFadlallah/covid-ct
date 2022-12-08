import math
import os
import numpy as np
import cv2
import torch

from dataset_utils import image_loader, image_from_array_loader
import utils
import convit_datasets


def calculate_metrics(pred_true_list):
    TP, FP, TN, FN = [0, 0, 0, 0]
    for couple in pred_true_list:
        if couple[0] == 'covid':
            if couple[1] == 'P':
                TP += 1
            else:
                FP += 1
        else:
            if couple[1] == 'N':
                TN += 1
            else:
                FN += 1
    # Calculate Metrics
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    my_accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("Specificity is : " + str(specificity))
    print("Sensitivity is : " + str(sensitivity))
    print("Accuracy is : " + str(my_accuracy))

    print("True neg: "+str(TN))
    print("True pos: "+str(TP))
    print("False neg: "+str(FN))
    print("False pos: "+str(FP))


class DatasetArgs:
    data_set: str = 'TestMosMed'
    sampling_ratio: float = 1.0
    nb_classes: int = 2
    data_path: str
    folder_name: str
    slices_names: list
    input_size: int = 224
    color_jitter: float = 0
    aa: str = 'rand-m9-mstd0.5-inc1'
    train_interpolation: str = 'bicubic'
    reprob: float = 0.25
    remode: str = 'pixel'
    recount: int = 1

def create_data_loader(root, folder_name, slices_names):
    args = DatasetArgs()
    args.data_path = root
    args.folder_name = folder_name
    args.slices_names = slices_names
    dataset, _ = datasets.build_dataset(is_train=False, args=args)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32,
        shuffle=False, num_workers=10,
        pin_memory=True, drop_last=False
    )
    return data_loader

# Predicting from images of a full scan
def convit_predict_scan(convit_model, num_of_images, root_dir):
    middle_slice = 26 # instead of middle slice we pick 2/3 slice
    slices_names = ["/img_"+str(middle_slice)+".jpg"]
    pred_true_list = []
    for i in range(math.floor(num_of_images/2)):
        slices_names.append("/img_"+str(middle_slice+(i+1))+".jpg")
        slices_names.append("/img_"+str(middle_slice-(i+1))+".jpg")
    print(f"selected slices to be predicted {slices_names}")

    scan_paths = os.listdir(root_dir)
    x = []
    y = []
    all = []
    for scan in scan_paths:
        votes = []
        for img_path in slices_names:
            path = root_dir + scan + img_path
            image = image_loader(path)
            output = convit_model(image)
            p = torch.argmax(output, dim=1)
### Zero is abnormal , 1 is normal scan
            ind = p.item()
            res = output[0][ind].item()
            # print(res)
            votes.append(res)
            # all.append(res)
            # print(f"prediction: {res}, true: {scan[0]}")
        x.append(votes)
        true_value = 1 if scan[0] == 'N' else 0
        y.append(true_value)
    return x, y

    #     majority = sum(votes) / num_of_images
    #     decision = 'normal' if majority > 0.3 else 'covid'
    #     # print(f"final decision for {scan} is {decision}")
    #     pred_true_list.append([decision, scan[0]])
    # calculate_metrics(pred_true_list)

def convit_predict_from_array(model, x_test, no_of_slices):
    middle_slice = 26 # instead of middle slice we pick 2/3 slice
    slices_names = [middle_slice]
    for i in range(math.floor(no_of_slices/2)):
        slices_names.append(middle_slice+i+1)
        slices_names.append(middle_slice-(i+1))
    print(f"selected slices to be predicted {slices_names}")

    votes = []
    pred_true_list=[]
    for i in range(len(x_test)):
        scan = np.moveaxis(x_test[i], -1, 0)
        for j in range(len(scan)):
            if j in slices_names:
                img_arr = scan[j]
                print(img_arr.shape)
                image = image_from_array_loader(img_arr)
                output = model(image)
                p = torch.argmax(output, dim=1)
    ### Zero is abnormal , 1 is normal scan
                votes.append(p)
        majority = sum(votes) / no_of_slices
        decision = 'normal' if majority > 0.5 else 'covid'
        y = 'P' if i < 50 else 'N'
        pred_true_list.append([decision, y])
    calculate_metrics(pred_true_list)

def cnn_predict_scan(model, root_dir):
    pred_true_list = []
    scan_paths = os.listdir(root_dir)
    for path in scan_paths:
        slices = []
        for i in range(40):
            img = cv2.imread(root_dir+path+"/"+'img_'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (144,144))
            normalized = img/255
            slices.append(normalized)
        scan = np.array(slices)
        scan = np.moveaxis(scan, 0, -1)
        prediction = model.predict(np.expand_dims(scan, axis=0))[0]
        score = [1 - prediction[0], prediction[0]]
        s = np.argmax(score, axis=0)
        # print(f"prediction for {path} is {s}")
        decision = 'normal' if s==0 else 'covid'
        print(f"decision: {decision}, path: {path[0]}")
        pred_true_list.append([decision, path[0]])
    calculate_metrics(pred_true_list)

def resnet_predict_single_scan(model, num_of_images, root_dir, scan):

    middle_slice = 26 # instead of middle slice we pick 2/3 slice
    slices_names = ["img_"+str(middle_slice)+".jpg"]
    pred_true_list = []
    for i in range(math.floor(num_of_images/2)):
        slices_names.append("img_"+str(middle_slice+(i+1))+".jpg")
        slices_names.append("img_"+str(middle_slice-(i+1))+".jpg")
    # print(f"selected slices to be predicted {slices_names}")
    path = root_dir + scan
    votes = []
    all = []
    for i in range(num_of_images):
        img = cv2.imread(path+"/"+slices_names[i])
        img = cv2.resize(img, (224,224))
        # normalized = img/255
        prediction = model.predict(np.expand_dims(img, axis=0))[0]
        score = [1 - prediction[0], prediction[0]]
        s = np.argmax(score, axis=0)
        votes.append(s)
        prob = score[s]
        all.append(prob)
    majority = sum(votes) / num_of_images
    decision = 'normal' if majority > 0.3 else 'covid'
    return [decision, scan[0]], all

def convit_predict_single_scan(convit_model, num_of_images, root_dir, scan):
    middle_slice = 26 # instead of middle slice we pick 2/3 slice
    slices_names = ["img_"+str(middle_slice)+".jpg"]
    pred_true_list = []
    for i in range(math.floor(num_of_images/2)):
        slices_names.append("img_"+str(middle_slice+(i+1))+".jpg")
        slices_names.append("img_"+str(middle_slice-(i+1))+".jpg")
    # print(f"selected slices to be predicted {slices_names}")

    votes = []
    all = []
    data_loader = create_data_loader(root_dir, scan, slices_names)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for images, target in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        for img in images:
            with torch.cuda.amp.autocast():
                output = convit_model(img.unsqueeze(0))
        
            p = torch.argmax(output, dim=1)
### Zero is abnormal , 1 is normal scan
            ind = p.item()
            res = output[0][ind].item()
            # print(p)
            # all.append(res/2) # divide by 2 so the range becomes 0,1 instead of 0,2
            all.append(ind)
            votes.append(ind)
            # print(f"prediction: {res}, true: {scan[0]}")
    true_value = 1 if scan[0] == 'N' else 0
    majority = sum(votes) / num_of_images
    decision = 'normal' if majority > 0.3 else 'covid'
    del data_loader
    return [decision, scan[0]], all

def cnn_predict_single_scan(model, root_dir, scan_name):
    pred_true_list = []
    path = root_dir + scan_name
    slices = []
    for i in range(40):
        img = cv2.imread(path+"/"+'img_'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (144,144))
        normalized = img/255
        slices.append(normalized)
    scan = np.array(slices)
    scan = np.moveaxis(scan, 0, -1)
    prediction = model.predict(np.expand_dims(scan, axis=0))[0]
    score = [1 - prediction[0], prediction[0]]
    s = np.argmax(score, axis=0)
    # print(f"prediction for {path} is {s}")
    decision = 'normal' if s==0 else 'covid'
    # print(f"decision: {decision}, path: {scan_name[0]}")
    prob = score[s]
    return [s, scan_name[0]], prob

