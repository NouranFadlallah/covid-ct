import tensorflow as tf
from tensorflow.keras import layers
from timm.utils import ModelEma, NativeScaler
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy

import utils

def create_cnn_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.7)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="3dcnn")
    model.load_weights("/content/drive/MyDrive/thesis/models/after_presentation/3d_mosmed254_exp14.h5")
    return model

def create_convit_model():
    model = create_model(
        "convit_mini",
        pretrained=False,
        num_classes=2,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        local_up_to_layer=2,
        locality_strength=0.5,
        embed_dim = 64,
    )
    device = torch.device('cpu')
    model.to(device)

    #Exponential moving average
    # model_ema = None
    # if args.model_ema:
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    linear_scaled_lr = 5e-4 * 32 * utils.get_world_size() / 512.0
    lr = linear_scaled_lr
    loss_scaler = NativeScaler()
    criterion = LabelSmoothingCrossEntropy()

    # if args.resume:
    PATH =  "/content/drive/MyDrive/thesis/models/convitMosMed/convit_88.pth"
    checkpoint = torch.load(PATH, map_location='cpu')    
    model_without_ddp.load_state_dict(checkpoint['model'])

    return model, device

def create_resnet_model():
    input_t = tf.keras.Input(shape=(224, 224, 3))
    res_model = tf.keras.applications.ResNet101(include_top=False,
                                        weights="imagenet",
                                        input_tensor=input_t)

    model = tf.keras.models.Sequential()
    model.add(res_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.load_weights("/content/drive/MyDrive/thesis/models/resnet101_vsconvit.h5")
    return model