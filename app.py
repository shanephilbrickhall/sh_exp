from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
import os

secret_key = os.environ.get('SECRET_KEY')
admin_user = os.environ.get('ADMIN_USERNAME')
admin_pass = os.environ.get('ADMIN_PASS')

app = Flask(__name__)
app.secret_key = secret_key


@app.route('/login', methods=['GET','POST'])
def user_login():
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    # print(POST_PASSWORD)

    if POST_USERNAME == admin_user and POST_PASSWORD == admin_pass:
        session['logged_in'] = True
    else:
        flash('Invalid Username or Password; you will be redirected, please revise login credentials and reattempt')
    return hello()

@app.route('/')
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    port = int(os.environ.get('PORT', 5000))
    app.run()