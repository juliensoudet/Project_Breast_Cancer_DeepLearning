import numpy as np
import pandas as pd

from tensorflow.keras import Input, Model, callbacks
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy




def data_augmentation(shape=(img_height, img_width, 3)):

    # Data Augmentation
    input_tensor=Input(shape=shape)
    x = layers.RandomFlip("horizontal")(input_tensor)
    x = layers.RandomZoom([0.1, 0.3])(x)
    x = layers.RandomTranslation(0.3, 0.3)(x)
    x = layers.RandomRotation([-1,1])(x)

    return input_tensor, x


def initialize_model(input_tensor,num_classes):


    # Load the pre-trained EfficientNetB0 model
    base_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_tensor=input_tensor)(x)

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Add custom classification head
    x = GlobalAveragePooling2D()(base_model)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    # Create the model
    model_eff = Model(inputs=input_tensor, outputs=output)

    return model_eff



def compile_model(model_eff):

    # Compile the model
    model_eff.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Display the model summary
    model_eff.summary()

    return model_eff



def train_model(model_eff, dataset_rgb, validation_dataset_rgb, epochs=50, batch_size=64):


    #This callback will save the model to a file after every epoch
    model_checkpoint = callbacks.ModelCheckpoint("model_best.keras", monitor='val_loss', verbose=0, save_best_only=True)

    #This callback reduces the learning rate when a metric has stopped improving.
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0)

    #This callback will stop the training if the monitored metric (in this case, validation loss) does not improve.
    early_stopper = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)


    # Train the model

    history_eff_improved= model_eff.fit(dataset_rgb,
                    validation_data=validation_dataset_rgb,
                    epochs=50,  batch_size=batch_size, callbacks=[model_checkpoint, lr_reducer, early_stopper]
                          )
    return history_eff_improved

def evaluate_model(model_eff,validation_dataset_rgb):

    # Evaluate the model
    loss, accuracy = model_eff.evaluate(validation_dataset_rgb)
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)

    return loss, accuracy
