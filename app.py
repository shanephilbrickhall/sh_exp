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

@app.route('/base_data_display', methods=['GET','POST'])
def base_data_display():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        base_data = pull_all_data()

        dates = base_data[0]
        iso_real_time = base_data[1]
        iso_day_ahead = base_data[2]
        avg_tmp_data = base_data[3]
        max_tmp_data = base_data[4]
        min_tmp_data = base_data[5]
        precip_data = base_data[6]
        print(iso_real_time,iso_day_ahead,avg_tmp_data,max_tmp_data,min_tmp_data,precip_data,dates)
        return render_template('base_data_chart.html', title='Base Data: ISONE RT, ISONE DA, BOST AVG TMP, '
                                                        'BOST HIGH TMP, BOST LOW TMP, BOST PRECIP',
                               max1=170, max2=80,max3=20,labels=dates,
                               values1=iso_real_time,values2=iso_day_ahead,values3=avg_tmp_data,
                               values4=max_tmp_data,values5=min_tmp_data,values6=precip_data )





if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run()