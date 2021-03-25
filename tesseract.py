import os.path
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import subprocess


def run_analysis(path):

    letters = {}

    # Loop over all training images
    for filename in tqdm(os.listdir(path)):
        word, level, ext = filename.split(".")

        for c in word:
            if c not in letters:
                letters[c] = 0
            letters[c] += 1

    print(letters.keys())
    return "".join(letters.keys())


def run_inference(path, letters):

    results = []

    # Loop over all training images
    for filename in tqdm(os.listdir(path)):
        id, level, ext = filename.split(".")

        command = "tesseract {source} stdout -c tessedit_char_whitelist={letters} quiet".format(
            source=os.path.join(path, filename),
            letters=letters
        )

        res = subprocess.run(command, stdout=subprocess.PIPE)
        res = res.stdout.decode("utf-8").split("\r")[0].strip().replace(" ", "")
        print(res)

        # Add your classification for this image
        results.append({"0": int(id), "1": "AVCD"})

    # 3 - Store your results ---------------------------------------------------
    df = pd.DataFrame(results)
    df.sort_values(by=df.columns[0], inplace=True)
    df.to_csv("submission.csv", header=False, index=False)


if __name__ == "__main__":

    # Constant variables
    training_set_path = "./data/train"
    test_set_path = "./data/test"

    # Top level functions
    letters = run_analysis(training_set_path)
    run_inference(test_set_path, letters)