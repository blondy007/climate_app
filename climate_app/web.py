"""Streamlit launcher with same-origin remembered-login endpoints."""
from climate_app.browser_auth import install_routes


def main():
    install_routes()
    from streamlit.web.cli import main as streamlit_main
    streamlit_main()


if __name__ == "__main__":
    main()
