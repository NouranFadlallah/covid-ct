from models import create_cnn_model, create_convit_model, create_resnet_model
from evaluation_utils import cnn_predict_single_scan, convit_predict_single_scan


dir = "/content/drive/MyDrive/thesis/ct_data/TrainCNNonImages/"
# scan = "PCase1/"

## CNN Predictions
cnn_model = create_cnn_model(width=144, height=144, depth=40)

# Convit Predictions
convit_model, device = create_convit_model()
convit_model.eval()
imsize = 224
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

#Resnet predictions
resnet_model = create_resnet_model()

x = []
y = []

x_test = []
y_test = []
for scan in os.listdir(dir):
    scan_num = scan[5:]
    cnn_result, cnn_prob = cnn_predict_single_scan(cnn_model, dir, scan)
    # convit_result, convit_prob = resnet_predict_single_scan(resnet_model, 11, dir, scan+"/")
    convit_result, convit_prob = convit_predict_single_scan(convit_model, 11, dir, scan+"/")
    print(f"{scan_num}: convit {convit_result[0]}, cnn {1 - cnn_result[0]}")

    # print(f"cnn result: {cnn_result}, convit: {convit_result}")
    # print(f"cnn prob: {cnn_prob}, decision: {cnn_result}")
    # print(f"convit prob: {convit_prob}, decision: {convit_result}")
    convit_prob.append(1 - cnn_result[0]) # because in convit 0 is covid but in cnn 0 is normal
    true_value = 1 if scan[0] == 'N' else 0

    if int(scan_num) <= 204 or int(scan_num) in range(255,459):
      x.append(convit_prob)
      y.append(true_value)
    else:
      x_test.append(convit_prob)
      y_test.append(true_value)

# print(np.asarray(x).shape)
# print(np.asarray(y).shape)