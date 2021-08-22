#import relevant libraries for flask, html rendering and loading the #ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import Model as m
app = Flask(__name__)

#loading the model and the preprocessor
model = pickle.load(open('model.pkl', 'rb'))
#std = pickle.load(open(‘std.pkl’,’rb’))

#Index.html will be returned for the input
@app.route('/')
def start():
   #if request.method == 'POST':
    #user_name = request.form["fname"]
    
 return render_template('index.html')

#predict function, POST method to take in inputs

@app.route('/predict',methods=['POST'])
def predict():
#take inputs for all the attributes through the HTML form
 if request.method == 'POST':
    user_name = request.form["fname"]
    prod_pred=m.prod_predict(user_name)
    if len(prod_pred)==0:
      return render_template('Nodata.html',result=user_name)
    else:
      return render_template('result.html',result=user_name , mk=prod_pred)
 #prediction=model.predict(user_name)
 #output='{0:.{1}f}'.format(prediction[0][1], 2)
 #output_print = str(float(output)*100)+'%'
 #print(output_print)
 #return render_template('index.html',output_print)
 #Recommend the products to the user Name
 #if float(output)>0.5:
 #
 #else:
 #
if __name__ == '__main__':
 app.run(debug=True)