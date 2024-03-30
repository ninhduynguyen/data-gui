import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import plotly.graph_objs as go
import pickle
from datetime import datetime,date

# 1. Read data
data_rfm_1 = pd.read_csv('OnlineRetail_mark4.csv')

# 2. load model
kmean_model = "kmean_model.pkl"
pkl_min_max_scaler = "min_max_scaler.pkl"

with open(pkl_min_max_scaler, 'rb') as file:  
    min_max_scaler = pickle.load(file)

with open(kmean_model, 'rb') as file:  
    kmean_model = pickle.load(file)

# 3. hàm xử lý dữ liệu dạng file đầu vào
def preprocess_data_input(data):
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.reset_index()
    data['InvoiceDatetime'] = None
    data['Monetary'] = None
    for i in range (data.shape[0]):
        thoigian = data.loc[i,'InvoiceDate']
        thoigian_1 = datetime.strptime(thoigian,"%d-%m-%Y %H:%M")
        data.loc[i,'InvoiceDatetime'] = thoigian_1
        data.loc[i,'Monetary'] = data.loc[i,'Quantity']*data.loc[i,'UnitPrice']
    data['Monetary'] = data['Monetary'].astype('float64')
    data['InvoiceDatetime'] = pd.to_datetime(data['InvoiceDatetime'])

    data = data[data['UnitPrice'] >= 0]
    
    data_temp = data[data.UnitPrice==0]
    data_temp = set(data_temp.InvoiceNo)
    data_temp = list(data_temp)
    data_temp = data[data['InvoiceNo'].isin(data_temp)]
    data_temp = data_temp.sort_values(by=['InvoiceNo','InvoiceDatetime'])
    data_temp = data_temp.groupby('InvoiceNo')['InvoiceNo'].count()
    data_temp = data_temp.to_frame()
    temp_1 = list(data_temp.index)
    temp_2 = list(data_temp.InvoiceNo)
    data_temp_1 = pd.DataFrame(
        {'InvoiceNo':temp_1,
         'InvoiceNo_ItemNum': temp_2}
    )
    data_temp_1 = data_temp_1[data_temp_1['InvoiceNo_ItemNum']==1]
    data = data[~data['InvoiceNo'].isin(data_temp_1['InvoiceNo'])]

    list_customer = set(data.CustomerID)
    data_pre_process = pd.DataFrame(columns=data_temp.columns)
    for customer in list_customer:
        data_temp = data[data['CustomerID']==customer]
        data_temp = data_temp.sort_values(by=['InvoiceDatetime'], ascending=True)
        list_ind = []
        for ind, val in data_temp.iterrows():
            if val['Quantity'] > 0:
                break
        else:
            list_ind.append(ind)
        if list_ind != []:
            data_temp = data_temp[~data_temp.index.isin(list_ind)]
        else:
            data_temp = data_temp
       
        for stockcode in set(data_temp.StockCode):
            data_temp_1 = data_temp[data_temp['StockCode']==stockcode]
            list_ind_1 = []
            for ind, val in data_temp_1.iterrows():
                if val['Quantity'] > 0:
                    break
                else:
                    list_ind_1.append(ind)
            if list_ind_1 != []:
                data_temp_1 = data_temp_1[~data_temp_1.index.isin(list_ind_1)]
            else:
                data_temp_1 = data_temp_1
            data_temp_2 = data_temp_1[data_temp_1['Quantity']>0]
            data_temp_3 = data_temp_1[data_temp_1['Quantity']<0]
            if data_temp_1['Quantity'].sum()<=0:
                data_pre_process=data_pre_process
            else:
                remain = data_temp_3['Quantity'].sum()
                for j in reversed(data_temp_2.index):
                    remain = data_temp_2.loc[j,"Quantity"] + remain
                    data_temp_2.loc[j,"Quantity"] = remain
                data_temp_2 = data_temp_2[data_temp_2['Quantity']>0]
                data_pre_process = pd.concat([data_pre_process,data_temp_2])    

    data_pre_process['Quantity'] = data_pre_process['Quantity'].astype('int64')
    max_date = date.today()
    Recency = lambda x: (max_date - x.max().date()).days
    Fequency = lambda x: len(x.unique())
    Monetary = lambda x: round(sum(x),2)
    data_rfm = data_pre_process.groupby('CustomerID').agg({
        'InvoiceDatetime':Recency, 
        'InvoiceNo':Fequency, 
        'Monetary':Monetary})
    data_rfm.columns = ['Recency','Frequency','Monetary']
    data_rfm = data_rfm.sort_values('Monetary',ascending=False)

    data_rfm = data_rfm.reset_index(drop = False)
    return data_rfm

#4. GUI

st.title("Xử lý cho Customer ")
menu = ["Home", "GUI for Customer Segmentation"]
choice = st.sidebar.selectbox('Danh mục', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice=='GUI for Customer Segmentation':
    st.write("##### 1. Upload lịch sử giao dịch trích xuất từ hệ thống")
    type = st.radio("Chọn cách nhập thông tin khách hàng", options=["Truy xuất theo 1 khách hàng sẵn có",
                                                                    "Upload thông tin khách hàng vào dataframe"])
    if type == "Truy xuất theo 1 khách hàng sẵn có":
        st.subheader("Nhập mã khách hàng")
        customer_id = st.text_input("Nhập mã khách hàng")
        st.write("Mã khách hàng:", customer_id)
        customer_id_entered = False
        if customer_id:
            customer_id_entered = True
        if customer_id_entered:
            data_rfm_1 = pd.read_csv('OnlineRetail_mark4.csv')
            customer_id = int(customer_id)
            position = data_rfm_1[data_rfm_1['CustomerID'] == customer_id].index
            st.write("vị trí dự báo nằm tạo index số", position)
            data_rfm_1 = data_rfm_1.drop(columns='CustomerID')
            features_to_scale = ['Recency', 'Frequency','Monetary']
            for feature in features_to_scale:
                data_rfm_1[[feature]] = min_max_scaler.fit_transform(data_rfm_1[[feature]])
            st.write("Dữ liệu đầu sau khi scale:")
            st.write(data_rfm_1.head(5))
            y_kmeans = kmean_model.fit_predict(data_rfm_1)
            data_rfm_1['cluster_5'] = pd.DataFrame(y_kmeans)
            st.write("Dữ liệu đầu sau khi dự báo:")
            st.write(data_rfm_1.loc[pd.Index(position),:])
            st.write("Khách hàng thuộc phân cụm số ",data_rfm_1.iloc[pd.Index(position),3].item())
            cluster = data_rfm_1.iloc[pd.Index(position),3].item()
            if cluster == 0:
                st.write("khách thuộc nhóm khách hàng yếu")
            elif cluster == 1:
                st.write("khách thuộc nhóm khách hàng trung bình yếu")
            elif cluster == 2:
                st.write("khách thuộc nhóm khách hàng trên trung bình")
            elif cluster == 3:
                st.write("khách thuộc nhóm khách hàng mạnh và mua nhiều")
            else:
                st.write("khách thuộc nhóm khách hàng mua nhiều nhất")
 
        
    elif type == "Upload thông tin khách hàng vào dataframe":
        st.subheader("File Uploader")
        file = st.file_uploader("Upload your file", type=["csv"])
        if file is not None:
            data = pd.read_csv(file)
            st.write(data.head(5))
            data_rfm_1 = preprocess_data_input(data)
            st.write("Dữ liệu đầu vào cho xử lý:")
            st.write(data_rfm_1.head(5))
            data = pd.read_csv('OnlineRetail_mark4.csv')
            data_rfm_1 = pd.concat([data_rfm_1,data])
            st.write("Dữ liệu ghép vào cho xử lý:")
            st.write(data_rfm_1.head(5))
            data_rfm_1 = data_rfm_1.drop(columns='CustomerID')
            
            features_to_scale = ['Recency', 'Frequency','Monetary']
            for feature in features_to_scale:
                data_rfm_1[[feature]] = min_max_scaler.fit_transform(data_rfm_1[[feature]])
            st.write("Dữ liệu đầu sau khi scale:")
            st.write(data_rfm_1.head(5))
            y_kmeans = kmean_model.fit_predict(data_rfm_1)
            data_rfm_1['cluster_5'] = pd.DataFrame(y_kmeans)
            st.write("Dữ liệu đã xữ lý xong:")
            st.write(data_rfm_1)
            st.write("Khách hàng thuộc phân cụm số ",data_rfm_1.iloc[0,3].item())
            cluster = data_rfm_1.iloc[0,3].item()
            if cluster == 0:
                st.write("khách thuộc nhóm khách hàng yếu")
            elif cluster == 1:
                st.write("khách thuộc nhóm khách hàng trung bình yếu")
            elif cluster == 2:
                st.write("khách thuộc nhóm khách hàng trên trung bình")
            elif cluster == 3:
                st.write("khách thuộc nhóm khách hàng mạnh và mua nhiều")
            else:
                st.write("khách thuộc nhóm khách hàng mua nhiều nhất")