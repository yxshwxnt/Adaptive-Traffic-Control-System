from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS,cross_origin
import time

app = Flask(__name__)

isAuthenticated=False 

@app.route('/')
def login_page():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST': 
        username = request.form['username']
        password = request.form['password']
        print(username,password)
        if username == 'abc' and password == '123':   
           isAuthenticated=True
           return redirect(url_for('predict_page'))
    return render_template('login.html')
    
# Define the function that calculates the time limits per vehicle
def calculate_time_limits(vehicles, base_timer, time_limits):
    time_limits_per_vehicle = []
    for vehicle, quantity in vehicles.items():
        time_limit = (quantity / sum(vehicles.values())) * base_timer
        if time_limits[0] < time_limit < time_limits[1]:
            time_limits_per_vehicle.append(time_limit)
        else:
            closest_limit = min(time_limits, key=lambda x: abs(x - time_limit))
            time_limits_per_vehicle.append(closest_limit)
    return time_limits_per_vehicle, sum(time_limits_per_vehicle)


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        # Get input values from the form
        signal1 = int(request.form['signal1'])
        signal2 = int(request.form['signal2'])
        signal3 = int(request.form['signal3'])
        signal4 = int(request.form['signal4'])
        # Define the inputs for the time limit calculation function
        vehicles = {'vehicle1': signal1, 'vehicle2': signal2, 'vehicle3': signal3, 'vehicle4': signal4}
        base_timer = 120
        time_limits = [5, 40]
        # Calculate the time limits per vehicle
        time_limits_per_vehicle, time_limits_sum = calculate_time_limits(vehicles, base_timer, time_limits)
        return render_template('Predict.html', time_limits_per_vehicle=time_limits_per_vehicle, time_limits_sum=time_limits_sum)
    # Render the template with default values
    return render_template('Predict.html', time_limits_per_vehicle=[], time_limits_sum=0)

@app.route('/index',methods=['GET','POST']) 
def vehicle_cnt():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)