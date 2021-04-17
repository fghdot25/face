import argparse
import os

import joblib
import numpy as np
from PIL import Image
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms

from extract_face_features import FaceFeaturesExtractor
from face_recogniser import FaceRecogniser
from preprocessing import ExifOrientationNormalize
from settings import MODEL_DIR, USER_IMAGES_FOLDER



def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def load_data(features_extractor):
    dataset = datasets.ImageFolder(USER_IMAGES_FOLDER)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)

    return embeddings, labels, dataset.class_to_idx


def train(embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    softmax.fit(embeddings, labels)

    return softmax


def main():
    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(features_extractor)
    clf = train(embeddings, labels)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    main()
