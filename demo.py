import os.path
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import keras as k
import sklearn.preprocessing
import pickle


# Let's code !!

def run_training(path):
    X = []
    Y = []

    # Loop over all training images
    for filename in tqdm(os.listdir(path)):
        word, level, ext = filename.split(".")

        # Skip level2 and level3
        if level != "level1": continue

        # 1 - Load current image -----------------------------------------------
        image = cv2.imread(os.path.join(path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2 - Separate the letters ---------------------------------------------
        for i, letter in enumerate(word):
            X.append(image[0:35, (i * 20):((i + 1) * 20)].flatten())
            Y.append(letter)

    # 3 - Learn from this dataset X Y ------------------------------------------
    X = np.array(X, dtype=np.float) / 255.0  # Normalization
    Y = np.array(Y)

    # Transform letter in a binary vector representation
    lb = sklearn.preprocessing.LabelBinarizer().fit(Y)
    Y = lb.transform(Y)
    nb_labels = len(lb.classes_)

    # Store labels
    with open("./model/labels.dat", "wb") as f:
        pickle.dump(lb, f)

    model = k.models.Sequential()
    model.add(k.layers.Dense(128, input_dim=(20 * 35), activation='relu'))
    model.add(k.layers.Dense(nb_labels, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, Y, validation_split=0.2, batch_size=128, epochs=20, verbose=1)

    # 4 - store you model ------------------------------------------------------
    model.save("./model/model_v1.hdf5")


def run_inference(path):

    model = k.models.load_model("./model/model_v1.hdf5")
    with open("./model/labels.dat", "rb") as file:
        labels = pickle.load(file)

    results = []

    # Loop over all test images
    for filename in tqdm(os.listdir(path)):
        id, level, ext = filename.split(".")

        # Skip level2 and level3
        if level != "level1": continue

        # 1 - Load current image -----------------------------------------------
        image = cv2.imread(os.path.join(path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        X = []
        # 2 - Separate the letters ---------------------------------------------
        for i in range(4):
            X.append(image[0:35, (i * 20):((i + 1) * 20)].flatten())

        X = np.array(X, dtype=np.float) / 255.0 # Normalization

        prediction = model.predict(X)
        r = "".join(labels.inverse_transform(prediction))

        # Add your classification for this image
        results.append({"0": int(id), "1": r})

    # 3 - Store your results ---------------------------------------------------
    df = pd.DataFrame(results)
    df.sort_values(by=df.columns[0], inplace=True)
    df.to_csv("submission.csv", header=False, index=False)

if __name__ == "__main__":

    # Constant variables
    training_set_path = "./data/train"
    test_set_path = "./data/test"

    # Top level functions
    run_training(training_set_path)
    run_inference(test_set_path)
