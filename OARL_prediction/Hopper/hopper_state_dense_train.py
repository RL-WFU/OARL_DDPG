#LIBRARIES
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.losses import kullback_leibler_divergence
from keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix,accuracy_score

state_length = 11
action_length = 3

#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../../Datasets/Hopper_DDPG_transition.npy", allow_pickle=True)

#flattens and unpacks the np arrays
pre = np.concatenate(input[:,0]).ravel()
pre = np.reshape(pre, (pre.shape[0]//state_length,state_length))
action = np.concatenate(input[:,1]).ravel()
action = np.reshape(action, (action.shape[0]//action_length,action_length))
post = np.concatenate(input[:,2]).ravel()
post = np.reshape(post, (post.shape[0]//state_length,state_length))
done = np.concatenate(input[:,3]).ravel()
done = np.reshape(done, (done.shape[0]//1,1))

#re-concatenates them
data = np.column_stack((pre,action,post,done))

inputX = data[:,:state_length].astype('float64')
inputY = data[:,state_length+action_length:(2*state_length)+action_length].astype('float64')


trainX = inputX[:80000]
trainY = inputY[:80000]
valX = inputX[80000:]
valY = inputY[80000:]



es = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=50)

# design network
model = Sequential()
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(valY.shape[1]))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# fit network
history = model.fit(trainX, trainY, epochs=5000, batch_size=5000, verbose=2,validation_data = (valX,valY),shuffle=False, callbacks=[es])

model.save('Hopper_State_Dense.keras')
print(model.summary())

np.save("history_Hopper_State_Dense.npy", history.history, allow_pickle=True)