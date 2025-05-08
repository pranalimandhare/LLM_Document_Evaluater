import streamlit as st
import mysql.connector
from mysql.connector import Error
import bcrypt



def performLogout():
    st.session_state.logged_in = False


def checkCredentials(username, password):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="pranali123",
            database="users",
            port=3306 
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user:
            st.session_state["username"] = username
            st.session_state["password"] = password
            st.session_state["logged_in"] = True
            print("Success")
            return bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8'))
            

    except Error as e:
        print(f"Error: {e}")
        return None


if 'logged_in' in st.session_state and (st.session_state.logged_in == True):
    st.title(f"Welcome {st.session_state.username}!")
    st.button("Logout", on_click=performLogout)

else:
    with st.form('Login'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if checkCredentials(username, password):
                st.success("Logged In Successfully")
                st.switch_page("pages/main.py")
            else:
                st.error("Invalid Login Credentials")
