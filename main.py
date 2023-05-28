#MACHINE LEARNING

#1 Import the data
#2 Clean the data
#3 Split data. Training set/Test Set
#4 Create a Model
#5 Check the output
#6 Improve

#Tools:
#NumPy - list, arrays
# Pandas - csv
#scikit-learn - create a model
#matplotlib - visualize data
#jupyter notebooks -
#kaggle - datasets for free

from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

prediction = ImageClassification()
# SqueezeNet model also no longer exists, now the fastest is MobileNetV2
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(exec_path, 'house.jpg'), result_count=5)
for eachPred, eachProb in zip(predictions, probabilities):
    print(f'{eachPred} : {eachProb}')
