import uvicorn
from fastapi import FastAPI
from features import Features
import pickle

instance =FastAPI()
pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

@instance.get('/')
def message():
    return {'Hello, Welcome to the deployment section of Diabetes Prediction model'}

@instance.post('/predict')
def predict_diabetes(data:Features):
    data = data.dict()
    Pregnancies = data["Pregnancies"]
    Glucose = data["Glucose"]
    BloodPressure = data["BloodPressure"]
    SkinThickness = data["SkinThickness"]
    Insulin = data["Insulin"]
    BMI = data["BMI"]
    DiabetesPedigreeFunction = data["DiabetesPedigreeFunction"]
    Age = data["Age"]
    prediction = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if(prediction[0]>0.5):
        prediction = "diabetic"
    else:
        prediction="not diabetic"
    return{
    'prediction':prediction
    }
    if __name__ == '__main__':
        uvicorn.run(instance,host='192.168.0.107',port=8000)

    #uvicorn deployment:instance --reload

