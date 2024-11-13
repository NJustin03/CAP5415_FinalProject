import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load class labels and annotations
    test_img_dir = 'cars_test_labels.csv'
    test_imgs = pd.read_csv(test_img_dir)
    class_labels = test_imgs['true_class_name']

    # Display the first 5 images with their make and model
    img_dir = 'cars_test'
    images = os.listdir(img_dir)

    for i, img_name in enumerate(images[:5]):
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path)

        plt.imshow(image)
        plt.title(f"Label: {class_labels[i], img_name}")
        plt.axis('off')
        plt.show()