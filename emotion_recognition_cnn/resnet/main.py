from glob import glob

from keras import Model
from keras.applications.resnet_v2 import ResNet152V2
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator


def get_data(train_path: str, test_path: str, shape: list, batch_size: int):
    image_gen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        shear_range=10,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20,
    )

    # create generators
    train_generator = image_gen.flow_from_directory(
        train_path,
        target_size=shape,
        shuffle=True,
        batch_size=batch_size,
    )

    test_generator = image_gen.flow_from_directory(
        test_path,
        target_size=shape,
        shuffle=True,
        batch_size=batch_size,
    )

    return (
        glob(train_path + "/*/*.jp*g"),
        glob(test_path + "/*/*.jp*g"),
        train_generator,
        test_generator,
    )


def create_model(shape: list, nbr_classes: int):
    vgg = ResNet152V2(
        input_shape=shape + [3],
        weights="imagenet",
        include_top=False,
    )

    # Freeze existing VGG already trained weights
    for layer in vgg.layers:
        layer.trainable = False

    # get the VGG output
    out = vgg.output

    # Add new dense layer at the end
    x = Flatten()(out)
    x = Dense(nbr_classes, activation="softmax")(x)

    model = Model(inputs=vgg.input, outputs=x)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # model.summary()

    return model


if __name__ == "__main__":
    shape = [224, 224]
    nbr_classes = 7
    train_path = "../../Databases/FER-2013/train"
    test_path = "../../Databases/FER-2013/test"
    batch_size = 8
    epochs = 10

    train_files, test_files, train_generator, test_generator = get_data(
        train_path=train_path, test_path=test_path, shape=shape, batch_size=batch_size
    )

    model = create_model(shape=shape, nbr_classes=nbr_classes)

    early_stop = EarlyStopping(monitor="val_loss", patience=2)
    r = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        steps_per_epoch=len(train_files) // batch_size,
        validation_steps=len(test_files) // batch_size,
        callbacks=[early_stop],
    )

    score = model.evaluate_generator(test_generator)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save("savedModel")
