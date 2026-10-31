BANDS = [(9.0, "Good"), (35.4, "Moderate"), (55.4, "Unhealthy for sensitive groups"),
         (125.4, "Unhealthy"), (225.4, "Very unhealthy")]


def pm25_category(v):
    """US EPA PM2.5 (24 h) categories, 2024 breakpoints."""
    if v != v:
        return "n/a"
    for hi, name in BANDS:
        if v <= hi:
            return name
    return "Hazardous"

COLORS = {"Good": "#00e400", "Moderate": "#ffff00", "Unhealthy for sensitive groups": "#ff7e00",
          "Unhealthy": "#ff0000", "Very unhealthy": "#8f3f97", "Hazardous": "#7e0023"}


def aqi_category(v):
    """US AQI (0-500) categories."""
    if v != v:
        return "n/a"
    for hi, name in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
                     (200, "Unhealthy"), (300, "Very unhealthy")]:
        if v <= hi:
            return name
    return "Hazardous"
