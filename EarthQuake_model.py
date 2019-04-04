
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train_df_first=pd.read_csv('data/train.csv')
test_df=pd.read_csv('data/test.csv')
building_ownership_df_first=pd.read_csv('data/Building_Ownership_Use.csv')
building_structure_df_first=pd.read_csv('data/Building_Structure.csv')


train_df_first['has_repair_started']=train_df_first['has_repair_started'].fillna(0)
building_ownership_df_first['count_families']=building_ownership_df_first['count_families'].fillna(1)
building_ownership_df_first['has_secondary_use']=building_ownership_df_first['has_secondary_use'].fillna(0)

building_structure_df_first=building_structure_df_first.replace({'condition_post_eq':{'Damaged-Repaired and used':'Damaged Repaired and used','Damaged-Rubble unclear':'Damaged Rubble unclear','Damaged-Rubble Clear-New building built':'Damaged Rubble Clear New building built','Damaged-Not used':'Damaged Not used','Damaged-Rubble clear':'Damaged Rubble clear','Damaged-Used in risk':'Damaged Used in risk'}})
building_structure_df_first=building_structure_df_first.replace({'foundation_type':{'Mud mortar-Stone/Brick':'Mud mortar Stone or Brick','Cement-Stone/Brick':'Cement Stone or Brick','Bamboo/Timber':'Bamboo or Timber'}})
building_structure_df_first=building_structure_df_first.replace({'roof_type':{'Bamboo/Timber-Light roof':'Bamboo or Timber Light roof','RCC/RB/RBC':'RCC or RB or RBC','Bamboo/Timber-Heavy roof':'Bamboo or Timber Heavy roof'}})
building_structure_df_first=building_structure_df_first.replace({'ground_floor_type':{'Brick/Stone':'Brick or Stone'}})
building_structure_df_first=building_structure_df_first.replace({'other_floor_type':{'TImber/Bamboo-Mud':'TImber or Bamboo Mud','Timber-Planck':'Timber Planck','RCC/RB/RBC':'RCC or RB or RBC'}})
building_structure_df_first=building_structure_df_first.replace({'position':{'Attached-1 side':'Attached 1 side','Attached-2 side':'Attached 2 side','Attached-3 side':'Attached 3 side'}})
building_structure_df_first=building_structure_df_first.replace({'plan_configuration':{'L-shape':'L shape','T-shape':'T shape','U-shape':'U shape','Multi-projected':'Multi projected','E-shape':'E shape','H-shape':'H shape'}})

building_structure_df_first['position']=building_structure_df_first['position'].fillna('Not attached')
building_structure_df_first['plan_configuration']=building_structure_df_first['plan_configuration'].fillna('Rectangular')
test_df['has_repair_started']=test_df['has_repair_started'].fillna(0)
## Merge all other datasets with train and test data
#merge two building datasets on common column i.e building_id
buildings_other_df=pd.merge(building_ownership_df_first,building_structure_df_first,on="building_id")
final_test_df=pd.merge(test_df,buildings_other_df,how="left",on="building_id")
building_id=final_test_df['building_id']
final_test_df=final_test_df.drop('building_id',axis=1)
final_train_df=pd.merge(train_df_first,buildings_other_df,how="left",on="building_id")

damage_grade=pd.DataFrame(final_train_df.damage_grade)
damage_grade_one_hot=pd.get_dummies(damage_grade)
#drop building id and damage grade from the train data
final_train_df=final_train_df.drop('building_id',axis=1)
final_train_df=final_train_df.drop('damage_grade',axis=1)

final_train_encoded_features=pd.get_dummies(final_train_df)
final_test_encoded_features=pd.get_dummies(final_test_df)
final_train, final_test = final_train_encoded_features.align(final_test_encoded_features, join='left', axis=1)

encoded=list(final_train.columns)
print('{} No of columns in train after one hot encoding'. format(len(encoded)))

encoded_2=list(final_test.columns)
print('{} No of columns in test after one hot encoding'. format(len(encoded_2)))

scaler=MinMaxScaler()
final_scaled_train=pd.DataFrame(scaler.fit_transform(final_train))
final_scaled_train.columns=final_train.columns

final_scaled_test=pd.DataFrame(scaler.fit_transform(final_test))
final_scaled_test.columns=final_test.columns


final_scaled_train.to_csv('final_scaled_train.csv',index=False)
final_scaled_test.to_csv('final_scaled_test.csv',index=False)

X_train,X_test,y_train,y_test=train_test_split(final_scaled_train,damage_grade_one_hot,test_size=0.25)

# SIMPLE FEED-FORWARD NEURAL NETWORKS MODEL MADE USING KERAS HAVING 6 HIDDEN LAYERS AND ACTIVATION FUNCTION RELU in the FIRST 5 LAYERS WHILE SOFTMAX IN THE LAST
# TO DETERMINE OR PREDICT THE CLASSES OF OUR TRAINING DATA

keras_model= Sequential()

keras_model.add(Dense(256,input_shape=X_train.shape[1:],activation='relu'))
keras_model.add(Dense(128,activation='relu'))
keras_model.add(Dense(64,activation='relu'))
keras_model.add(Dense(32,activation='relu'))
keras_model.add(Dense(16,activation='relu'))
keras_model.add(Dense(5,activation='softmax'))

keras_model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

import time
epochs=10

checkpointer = ModelCheckpoint(filepath='keras_model_best_weights.hdf5', 
                               verbose=1, save_best_only=True)
start=time.time()

print("Training Keras Model:")

history= keras_model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=epochs,
                         callbacks=[checkpointer], verbose=1)

end=time.time()

total_time=end-start
print("Time Taken(in Minutes) to Train the Model:", total_time/60)
keras_model.load_weights('keras_model_best_weights.hdf5')

accuracy_1=keras_model.evaluate(x=X_test,y=y_test,batch_size=32)
print("Accuracy of Keras model : ",accuracy_1[1])

model_json = keras_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)




