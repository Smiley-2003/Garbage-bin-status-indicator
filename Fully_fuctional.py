#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import cv2
import seaborn
import colorama
from colorama import Fore
import numpy as np
import smtplib as sm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[7]:


cam = cv2.VideoCapture(0)
cv2.namedWindow("capture")
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("capture", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 13:
        # ENTER pressed
        img_name = "dustbin{}".format(img_counter)
        cv2.imwrite("D:\\Project\\training and classification\\Testing\\"+img_name+".png", frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


# In[8]:


Training= ImageDataGenerator(rescale=1/255)
Validation = ImageDataGenerator(rescale=1/255)


# In[9]:


Training_dataset = Training.flow_from_directory('D:\\Project\\training and classification\\Training',
                                         target_size = (200,200),
                                         batch_size=3,
                                         class_mode = 'binary')

Validation_dataset = Validation.flow_from_directory('D:\\Project\\training and classification\\Validation',
                                                   target_size = (200,200),
                                                   batch_size =3,
                                                   class_mode = 'binary')


# In[10]:


Training_dataset.class_indices


# In[12]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation ='relu', input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    # layer2
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    # layers3
                                    tf.keras.layers.Conv2D(128,(3,3),activation ='relu',),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #Flatten
                                    tf.keras.layers.Flatten(),
                                    # Dense layer
                                    tf.keras.layers.Dense(512,activation= 'relu'),
                                    ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')                               
    
])


# In[13]:


model.compile(loss='binary_crossentropy',
             optimizer = RMSprop(learning_rate=0.001),
             metrics =['accuracy'])


# In[14]:


history=model.fit(Training_dataset,
                    steps_per_epoch =3,
                    epochs=25,
                    validation_data= Validation_dataset)


# In[71]:


dir_path ='D:\\Project\\training and classification\\Testing'
empty_count = 0
full_count = 0
predicted = []

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis =0)
    images = np.vstack([X])
    val=model.predict(images)
    if val==0:
        empty_count += 1
        print('Dustbin is Empty')
        predicted.append(0)
    else:
        full_count += 1
        print('Dustbin is full, Please Empty me')
        predicted.append(1)
        ob=sm.SMTP('smtp.gmail.com',587)
        ob.ehlo()
        ob.starttls()
        ob.login('ashishkr11989198@gmail.com','hklawccdlfqsmigu')
        subject="Muncipal Management"
        body="The dustbin of Moti bagh is almost FULL, Please send  team to vacant me"
        message="subject:{}\n\n{}".format(subject,body)
        listadd=['ashishkr119191@gmail.com',
                 'ashish119191@gmail.com']
        ob.sendmail('ashishkr119191@gmail.com',listadd,message)
        print(Fore.RED +"Mail sent")
        ob.quit()

        
        
predicted = [0 if val == 0 else 1 for val in predicted]
print("Number of empty dustbins detected: ",empty_count)
print("  And,Actual empty count is 6")      
print("Number of full dustbins detected: ",full_count)
print("  And Actual full count is 14")


# In[62]:


actual=[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


# In[67]:


print("Actual=",actual)
print("Predicted:",predicted)

cm = confusion_matrix(actual, predicted)
print("Confusion Matrix: \n", cm)

seaborn.heatmap(cm,cmap="Greens",annot=True,
                cbar_kws={"orientation":"vertical","label":"Color bar"},
              )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[64]:


plt.plot(history.history['accuracy'],color='red',label='Training')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation')
plt.title('Training and Validation accuracy of CNN models with THREE layers')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("D:\\Project\\training and classification\\Result and analysis\\CNN_accuracy.png")
plt.show()


# In[65]:


plt.plot(history.history['loss'],color='red',label='Training')
plt.plot(history.history['val_loss'], color='blue', label='Validation')
plt.title('Training and Validation loss of CNN models with THREE layers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("D:\\Project\\training and classification\\Result and analysis\\CNN_loss.png")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




