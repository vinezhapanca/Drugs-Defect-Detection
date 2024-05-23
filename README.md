# Drugs Defect Detection
Final Project ADSP 31009 Machine Learning and Predictive Analytics -  MS in Applied Data Science - University of Chicago



## 1. Background 

In drug manufacturing, maintaining high quality and safety standards is critical. However, current quality control methods rely largely on manual inspection, which is not only slow but also subject to human error.
The diversity of drug types and the subtlety of defects pose significant challenges for manual inspections. These methods are inefficient at predicting and preempting worsening conditions in the production process, leading to potential risks in drug safety and efficacy.
This project aims to develop an advanced drug defect detection system using deep learning model, specifically YOLOv8, and assessing its performance for potential integration into real-time production and quality assurance settings.

## 2. Why YOLOv8?

YOLOv8 is a cutting-edge, real-time detection model that integrates object detection and classification in a single operation. 
Its scalability and ability to benefit from transfer learning, even with a dataset currently limited to single objects, promise enhanced performance and readiness for more complex future scenarios. Though originally trained on the non-pharmaceutical COCO dataset, YOLOv8's versatile object recognition can be fine-tuned to our specifications. 
The Nano version, in particular, stands out for its speed and efficiency, making it an optimal choice for deployment on resource-constrained edge devices.

## 3. Dataset

There are 3 types of drugs in this dataset : Capsule, Tablet, and Softgel.
For each type, there are some normal drugs and some defect drugs.
For capsule and tablet, I found 2 interesting datasets, therefore I mixed them so that the model can learn from more various data. Unfortunately, I could not find another dataset for Softgel.

I roughly curated the dataset, ensuring that if an image classified as defect I can confidently validate that it is indeed defect. 

a. [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Capsule and Tablet)

b. [PillQC Dataset](https://github.com/matlab-deep-learning/pillQC) (Tablet)

c. [Sensum Solid Oral Dosage Forms](https://www.sensum.eu/sensumsodf-dataset/) (Capsule and Softgel)

After the preprocess, the distribution of the dataset looks like this :

![image](https://github.com/vinezhapanca/Drugs-Defect-Detection/assets/24844195/0bbbf4e9-9518-4cce-9a23-2dfab4815d93)

Some sample images and how many images with that class/label:

![image](https://github.com/vinezhapanca/Drugs-Defect-Detection/assets/24844195/9230ddbe-1ce5-428b-95df-a8a961e100aa)



## 4. Classification Model

Looking at the dataset, I decided to try building a classification model using simple Convolutional Neural Network. 
I uploaded it in "EDA and Simple CNN.ipynb"

Something I learned here :
1. We need to be careful on distributing the dataset on train, valid, and test set. If we only take random data from overall data, some classes might appear many times in one set and be less represented in other sets. This might happen especially if the classes are highly imbalanced. In this case, I tried to take random data proportional to each class, such that the problem doesn't happen.
2. We can use flow_from_directory function from keras, but for the test_generator I think it is better to set shuffle = False. We are usually interested to analyze the performance, for example by using confusion matrix, however if we set shuffle = True, the ground truth index will keep changing and the confusion matrix will not represent the actual performance of the model.   
```  
   test_generator = test_datagen.flow_from_directory(
    directory=r"./data_split/test/",
    target_size=(255, 255),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
)
``` 

## 5. Data Preparation for YOLOv8 model

In order to train YOLOv8 model, we will need the data to be in form of bounding box area, so we will need to do data annotation.
People usually refer to Roboflow to do data annotation, but in this opportunity I learned to use a open source tool called [label-studio](https://github.com/HumanSignal/label-studio/). 

It is an on-prem based tool, and the annotation is saved in our local. We still need to register email and password, however it can be accessed without internet access. 
It is also possible for collaboration, for example if it is installed in a server which can be accessed by some people. We just need to make separate account to each person. 
There is actually a limitation in terms of how many data can be uploaded from local, however we can upload the image files to Google Cloud Storage without any limitation, and sync to our label studio instance. You can follow the steps [here]. 

Something I learned here :
If possible, although the data might be in different folder, please name each file differently. There is an option save the annotation as YOLO format, however in my case the data was truncated, which I suspect due to my file naming issue. Nevertheless, I also needed some customization (related to what I said earlier about dividing data into train, val, and test), so I decided to save the annotation to COCO format and convert it to YOLO format. 
The basic code was from [here](https://github.com/ultralytics/JSON2YOLO) , and the final code can be seen at "COCO JSON to YOLO.ipynb"


## 6. Finetune YOLOv8 Nano



## 7. Hyperparameter tuning YOLOv8 model

I used [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to help with hyperparameter tuning.
Some hyperparameter that I tuned are batch_size, imgsz (image size), and lrf (learning rate final). I also implemented early stopping mechanism. 
Due to time and resource constraint, I only did 25 iteration for 30 epoch, and taking 3 samples (I used Bayesian Optimization Search Algorithm).
However, the result is quite promising, increasing overall Recall value to .......


## 8. Challenge : Generalization
