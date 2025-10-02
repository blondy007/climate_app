import streamlit as st


def handle_station_feedback() -> None:
    feedback = st.session_state.pop("add_station_feedback", None)
    if not feedback:
        return
    level, message = feedback
    if level == "success":
        st.success(message)
    elif level == "info":
        st.info(message)
    elif level == "error":
        st.error(message)
