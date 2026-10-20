"""Real-time Google data is confined to the authenticated owner's session."""

PRIVATE_PROVIDERS = {"weathernext3", "climateapp_private"}


def private_view():
    from climate_app.auth import authenticated
    return authenticated()


def visible_forecasts(frame):
    return frame if private_view() else frame[~frame.provider.isin(PRIVATE_PROVIDERS)].copy()
