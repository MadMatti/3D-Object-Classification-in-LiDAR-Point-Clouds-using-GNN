from preprocess import kitti as preprocess_kitti
import os 

DATASET_PATH = "/tmp_workspace/KITTI/"
DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "training")
DATASET_TEST_PATH = os.path.join(DATASET_PATH, "testing")
SAVE_PATH = os.path.join(DATASET_PATH, "processed")


def main():
    # Create the save path
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Process the training dataset
    preprocess_kitti.preprocess(DATASET_TRAIN_PATH, SAVE_PATH)


if __name__ == '__main__':
    main()
    