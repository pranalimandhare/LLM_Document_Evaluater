import streamlit as st


def setPages():
    if 'logged_in' in st.session_state and (st.session_state.logged_in == True):
        pageName = 'Logout'
    else:
        pageName = 'Login'
    login = st.Page("pages/login.py", title=pageName,
                    icon=":material/" + pageName.lower() + ":")
    profile = st.Page("pages/main.py", title="App",
                      icon=":material/dashboard:")
    
    pg = st.navigation([login, profile])
    pg.run()


setPages()
