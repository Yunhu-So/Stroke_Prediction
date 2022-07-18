# Stroke_Prediction
 
After exported stroke prediction dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), I have created a deep learning model that predicts the probability of a patient having a stroke based on several input parameters with **Multilayer Perception (MLP)** algorithm

## Attribute Information

1. Gender
   * Male = 0
   * Female = 1
   * Other = 2
2. Age
3. Hypertension
   * Doesn't have hypertenstion = 0
   * Have hypertension = 1
4. Heart Disease
   * Doesn't have heart diseases = 0
   * Have heart diseases = 1
5. Marital Status 
   * Not married = 0
   * Married = 1
6. Work Type
   * Private = 0
   * Self-employed = 1
   * Government Job= 2
   * Children = 3
   * Never Worked = 4
7. Residence
   * Urban = 0
   * Rural = 1
8. Average Glucose Level
9. BMI
10. Smoking Status

### One Hot Encoding 

I have imported OneHotEncoder from scikit-learn for column _Smoking Status_ since it was formally consisted of categorical values. 

After Implementation, a patient's smoking status is represented:
   * 1 0 0 = Formerly Smoked
   * 0 1 0 = Never Smoked
   * 0 0 1 = Smokes
   * 0 0 0 = Unknown or Unavailable Data
   
## Prediction

To predict the probability of a 40 year old male who has hypertension, has heart disease, is married, has a private job, lives at urban, has a glucose level of 170, has a bmi of 20, and smokes:

Type this on the code
```
prediction = model.predict([[0, 0, 1, 0, 40, 1, 1, 1, 0, 0, 170, 20]])
print(prediction)
```
First three numbers indicates the smoking status (it is now in the first three numbers due to one hot encoding).

Starting from the fourth number, it is in the same ordering like the Attribute section

To run, type 
```
$ python3 stroke_prediction.py
```
