from pyexpat import model
from keras.models import load_model
import pandas as pd
import numpy as np
test=pd.read_csv('publicgitcode\\mycode\\ai\\Handwritten-digit-recognitionV2.0 CNN\\test.csv')
long=len(test)
Test=test.values.reshape(long,28, 28, 1)
Test = Test / 255.0  #归一化
del test 
model=load_model('v1model.h5')
y_pred = np.array(model.predict(Test))

ylist=np.argmax(y_pred,axis=1)
i=np.arange(1,long+1)
dict = {'ImageId': i, 'Label': ylist}
result = pd.DataFrame(dict)
result.to_csv('result.csv',index = False)

