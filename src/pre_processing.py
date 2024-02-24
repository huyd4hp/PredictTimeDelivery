import os
import numpy as np
import pandas as pd 
import math 
import random 
from datetime import datetime
from datetime import timedelta

# Định dạng lại ngày giờ 
def date_time(date):
    date = str(date)
    if '23:60' in date:
        date = date.split(" ")
        date[0] = datetime.strptime(date[0], "%d-%m-%Y")
        date[1] = date[1].replace('23:60', "00:00")
        date[0] += timedelta(days=1)
        date[0] = str(date[0])
        date = date[0] + " " + date[1]
        date = date.replace('00:00:00', "")
        date = datetime.strptime(date, "%Y-%m-%d %H:%M")
        return date
    if '24:' in date:
        date = date.split(" ")
        date[0] = datetime.strptime(date[0], "%d-%m-%Y")
        date[1] = date[1].replace('24:', "00:")
        date[0] += timedelta(days=1)
        date[0] = str(date[0])
        date = date[0] + " " + date[1]
        date = date.replace('00:00:00', "")
        date = datetime.strptime(date, "%Y-%m-%d %H:%M")
        return date
    if ':60' in date:
        date = date.split(" ")
        date[0] = datetime.strptime(date[0], "%d-%m-%Y")
        date[0] = str(date[0])
        date[1] = date[1].replace('60', '00')
        date[1] = date[1].split(":")
        date[1] = str(int(date[1][0])+1) + ":" + date[1][1]
        date = date[0] + " " + date[1]
        date = date.replace('00:00:00', "")
        date = datetime.strptime(date, "%Y-%m-%d %H:%M")
        return date
    if '-' in date:
        date = date.split(" ")
        date[0] = datetime.strptime(date[0], "%d-%m-%Y")
        date[0] = str(date[0])
        date = date[0] + " " + date[1]
        date = date.replace('00:00:00', "")
        date = datetime.strptime(date, "%Y-%m-%d %H:%M")
        return date


# Tính khoảng thời gian giữa Time_order -> Time_order_picked
def time_range(df):
    time_range = []
    for i in range(len(df)):
        if '2022' in str(df['Time_order'][i]):
            time_range.append(df['Time_order_picked'][i] - df['Time_order'][i])
    return time_range


# Xóa hàng có nhiều giá trị NaN
def drop_missing_big_data(df):

    # Xóa những hàng có trên 4 giá trị NaN
    for i in range(len(df)):
        if sum(np.array(df[i:i+1].isnull().sum())) > 3:
            df = df.drop(df[i:i+1].index) 
    # Xóa những hàng chứa giá trị lỗi 
    df = df.drop(df[df['Restaurant_latitude']<=0].index)
    df = df.drop(df[df['Restaurant_longitude']<=0].index)
    df = df.drop(df[df['Delivery_location_latitude']<=0].index)
    df = df.drop(df[df['Delivery_location_longitude']<=0].index)
    df = df.drop(df[df['Delivery_person_ratings'] > 5].index)
    df["Multiple_deliveries"] = df["Multiple_deliveries"].replace(to_replace = 0, value= 1)
    # Tạo lại cột index
    df["Index"] = 0
    for i in range(len(df)):
        df["Index"][i:i+1] = i 
    df.set_index("Index", inplace = True)
    return df


# Tính khoảng cách giữa cửa hàng và nơi giao hàng 
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295;   
    c = math.cos
    a = 0.5 - c((lat2 - lat1) * p)/2 + c(lat1 * p) * c(lat2 * p) * (1 - c((lon2 - lon1) * p))/2

    return 12742 * math.asin(math.sqrt(a))


# Khởi tạo feature distance dựa trên 4 feature có sẵn gồm kinh độ/ vĩ độ của 2 địa điểm
def create_feature_distance(df):
    # Khởi tạo cột Distance có tất cả giá trị đều bằng 0
    df["Distance"] = 0

    for i in range(len(df)):
        df["Distance"][i:i+1] = distance(df["Restaurant_latitude"][i:i+1], df["Restaurant_longitude"][i:i+1],
                                        df["Delivery_location_latitude"][i:i+1],df["Delivery_location_longitude"][i:i+1])

    return df


# Chỉnh sửa 2 feature 'Time_order' và 'Time_order_picked'
def edit_time(df):
    df['Time_order'] = df['Order_date'] + ' ' + df['Time_order']
    df['Time_order_picked'] = df['Order_date'] + ' ' + df['Time_order_picked']
    for i in range(len(df)):
        df['Time_order'][i] = date_time(df['Time_order'][i])
        df['Time_order_picked'][i] = date_time(df['Time_order_picked'][i])
    return df

# Điền các giá trị khuyết: Có 8 feature chứa giá trị khuyết
def filling_missing_data(df):
    # Điền khuyết cho 7 feauture 
    df["Delivery_person_Age"] = df["Delivery_person_Age"].replace(to_replace = np.NaN, value= random.choice(np.array(df["Delivery_person_Age"])))
    df["Multiple_deliveries"] = df["Multiple_deliveries"].replace(to_replace = np.NaN, value= random.choice(np.array(df["Multiple_deliveries"])))
    df["Road_traffic_density"] = df["Road_traffic_density"].replace(to_replace = np.NaN, value= random.choice(np.array(df["Road_traffic_density"])))
    df["Weather_conditions"] = df["Weather_conditions"].replace(to_replace = np.NaN, value= random.choice(np.array(df["Weather_conditions"])))
    df["Delivery_person_ratings"] = df["Delivery_person_ratings"].replace(to_replace = np.NaN, value= random.choice(np.array(df["Delivery_person_ratings"])))
    df["Festival"] = df["Festival"].replace(to_replace = np.NaN, value= random.choice(np.array(df["Festival"])))
    df["City"] = df["City"].replace(to_replace = np.NaN, value= random.choice(np.array(df["City"])))

    # Điền khuyết cho feauture Time_order:
    timerange = time_range(df)

    for i in range(len(df)):
        if '2022' in str(df['Time_order'][i]):
            continue
        else:
            df['Time_order'][i] = df['Time_order_picked'][i] - random.choice(timerange)
    
    return df


# Chỉnh sửa sắp xếp lại dataframe 
def edit_dataframe(df):
    df = create_feature_distance(df)    # Khởi tạo feature distance dựa trên 4 feature có sẵn gồm kinh độ/ vĩ độ của 2 địa điểm
    df = edit_time(df)                  # Chỉnh sửa 2 feature 'Time_order' và 'Time_order_picked'  
    df = drop_missing_big_data(df)      # Xóa dữ liệu dư thừa và những hàng có nhiều giá trị NaN 
    df = filling_missing_data(df)       # Điền các giá trị khuyết

    # Xoá 4 feature 
    del df['Unnamed: 0']
    del df['ID']
    del df['Delivery_person_ID']
    del df['Order_date'] 

    # Để feautures Time_taken_(min) (Labels) ra cuối cùng 
    columns = df.columns.tolist()
    colums_new = columns[:-2] + columns[-1:] + columns[-2:-1]
    df = df[colums_new]

    return df


# Save file csv
def save_data_csv(df, path):
    df.to_csv(path)


# main
def main(df,path_save_new):
    df = edit_dataframe(df)             # Chỉnh sửa sắp xếp lại dataframe 
    save_data_csv(df, path_save_new)

