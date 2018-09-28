from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
from forecast_controller import *
import os

secret_key = os.environ.get('SECRET_KEY')
admin_user = os.environ.get('ADMIN_USERNAME')
admin_pass = os.environ.get('ADMIN_PASS')

app = Flask(__name__)
app.secret_key = secret_key


@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('site_directory.html')


@app.route('/login', methods=['GET','POST'])
def user_login():
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    # print(POST_PASSWORD)

    if POST_USERNAME == admin_user and POST_PASSWORD == admin_pass:
        session['logged_in'] = True
    else:
        flash('Invalid Username or Password; you will be redirected, please revise login credentials and reattempt')
    return home()


@app.route('/hello')
def hello():
    return "Hello World!"


@app.route('/base_data_display', methods=['GET'])
def base_data_display():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:

        # These lines are simply to show that this could use reactive graphics as opposed to static files
        # Graphics could be built using the Chart.js or Google Charts
        # As you can see this method simply leverages the code we examined earlier which loads and grooms the data
        # We can review the out put in the server logs for clarity

        base_data = pull_all_data()

        dates = base_data[0]
        iso_real_time = base_data[1]
        iso_day_ahead = base_data[2]
        avg_tmp_data = base_data[3]
        max_tmp_data = base_data[4]
        min_tmp_data = base_data[5]
        precip_data = base_data[6]
        print(iso_real_time,iso_day_ahead,avg_tmp_data,max_tmp_data,min_tmp_data,precip_data,dates)
        print("Below is base expirement data for display and includes the following: ISO-NE RT History, "
              "ISO-NE Day Ahead History, Boston Average Temp History, Boston High Temp History"
              "Boston Low Temp History, Boston Precipitation History, and the dates used for plotting them")
        return render_template('base_data_display.html')


@app.route('/model_comparison', methods=['GET'])
def model_comparison():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        # Again these lines are simply to show that this could use reactive graphics as opposed to static files
        # Graphics could be built using the Chart.js or Google Charts
        # As you can see the model is able to run on the remote server quickly and produce the expected results
        # We can review the out put in the server logs for clarity

        comp_model_data = compare_std_opt()

        dates = comp_model_data[0]
        std_model_prediction = comp_model_data[1]
        opt_model_prediction = comp_model_data[2]

        print("Below is the standard to grid search optimized comparison model run, output as follows: "
              "Standard Model Results, Optimized Model Results, and the dates used for plotting them")
        print(std_model_prediction,opt_model_prediction,dates)

        return render_template('model_comp_display.html')


@app.route('/scenario_comparison', methods=['GET'])
def scenario_comparison():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:

        # And again these lines are simply to show that this could use reactive graphics as opposed to static files
        # Graphics could be built using the Chart.js or Google Charts
        # As you can see this method also runs quickly and pulls in all the data sets and runs the optimized model
        # again producing the expected results we can review the out put in the server logs for clarity
        scenario_model_comp_data = scenario_build()

        dates = scenario_model_comp_data[0]
        iso_rt_hist_data = scenario_model_comp_data[1]
        avg_scn_data = scenario_model_comp_data[2]
        high_scn_data = scenario_model_comp_data[3]
        low_scn_data = scenario_model_comp_data[4]
        print("Below is the scenario comparison model run, output as follows: ISO-NE RT Historical Data, "
              "Average Temp Scenario, High Temp Scenario, Low Temp Scenario, and the dates used for plotting them")
        print(iso_rt_hist_data,avg_scn_data,high_scn_data,low_scn_data,dates)

        return render_template('hi_low_avg_scenario.html')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run()