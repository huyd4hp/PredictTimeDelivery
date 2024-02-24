import pre_processing
import pandas as pd
import os


def main():
    # Read .csv file
    df = pd.read_csv(os.path.abspath("../Predict_the_time_of_arrival_for_the_delivery_persons/dataset/dataset.csv"))
    # Clean data
    path_save_new = os.path.abspath("../Predict_the_time_of_arrival_for_the_delivery_persons/dataset/dataset_clean.csv")
    pre_processing.main(df, path_save_new)
    df = pd.read_csv(path_save_new)
    print(df)

    # EDA + deploy
    # Nhóm sẽ tiến hành trên file dataset_clean.csv

    # Model


if __name__ == '__main__':
    main()
