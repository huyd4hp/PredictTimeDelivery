import pandas as pd
import os


# Read file .txt
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()


# Create a list containing information about one .txt file
def information(text_information):
    list_information = []

    for i in range(20):
        text = text_information[i][27:]
        text = text.replace(' ', '')
        text = text.replace('\n', '')
        list_information.append(text)
    return list_information


# Create a list containing information about all .txt files
def merge_text_file(path):
    list_information_all = []
    path = os.path.abspath(path)

    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            list_information_all.append(read_text_file(file_path))
    return list_information_all


# Create features for dataframe
def create_feature():
    arr = []
    df = pd.DataFrame(arr, columns=['ID', 'Delivery_person_ID', 'Delivery_person_Age', 'Delivery_person_ratings',
                                    'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude',
                                    'Delivery_location_longitude', 'Order_date', 'Time_order', 'Time_order_picked',
                                    'Weather_conditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
                                    'Type_of_vehicle', 'Multiple_deliveries', 'Festival', 'City', 'Time_taken_(min)'])
    return df


# Create dataframe from all file texts
def create_data_csv(list_information_all):
    df = create_feature()

    for i in range(len(list_information_all)):
        df.loc[i] = information(list_information_all[i])
    return df


# Save file csv
def save_data_csv(df, path):
    df.to_csv(path)


def main():
    list_information_all = merge_text_file("../delivery-time-prediction/dataset/raw_data")
    df = create_data_csv(list_information_all)
    save_data_csv(df, os.path.abspath("../delivery-time-prediction/dataset/dataset.csv"))


if __name__ == '__main__':
    main()
