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


# --- PM2.5 -> AQI on two scales (used for the phone-style screens) ---
US_BP = [(0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]
IN_BP = [(0, 30, 0, 50), (30.1, 60, 51, 100), (60.1, 90, 101, 200), (90.1, 120, 201, 300), (120.1, 250, 301, 400), (250.1, 500, 401, 500)]
US_CATS = [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"), (200, "Unhealthy"), (300, "Very unhealthy"), (10 ** 9, "Hazardous")]
IN_CATS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very poor"), (10 ** 9, "Severe")]
IN_COLORS = {"Good": "#009a44", "Satisfactory": "#8cc63f", "Moderate": "#f4d03f", "Poor": "#f39c12", "Very poor": "#e74c3c", "Severe": "#8b0000"}
DARK_TEXT = {"Good": False, "Moderate": True, "Unhealthy for sensitive groups": True, "Satisfactory": True, "Poor": True}
ADVICE = {"Good": "Air is clean. Enjoy outdoor activities.",
          "Moderate": "Unusually sensitive people should reduce long, heavy outdoor exertion.",
          "Unhealthy for sensitive groups": "Children, older adults and people with heart or lung disease should limit prolonged outdoor exertion.",
          "Unhealthy": "Everyone should reduce prolonged outdoor exertion; sensitive groups should stay indoors.",
          "Very unhealthy": "Avoid outdoor exertion; sensitive groups stay indoors. A well-fitted N95/FFP2 mask helps if you must go out.",
          "Hazardous": "Stay indoors with windows closed, use an air purifier if you have one, and avoid outdoor activity.",
          "Satisfactory": "Minor breathing discomfort is possible for very sensitive people.",
          "Poor": "Breathing discomfort on prolonged exposure. Avoid heavy outdoor exertion; sensitive groups stay indoors.",
          "Very poor": "Respiratory illness is possible on prolonged exposure. Avoid outdoor exertion; wear a well-fitted N95/FFP2 mask if you must go out.",
          "Severe": "Affects healthy people too. Stay indoors with windows closed and avoid outdoor activity."}
ADVICE["Moderate_IN"] = "People with lung or heart disease, children and older adults may feel breathing discomfort."


def _interp(pm, bps):
    pm = max(0.0, float(pm))
    for lo, hi, a, b in bps:
        if pm <= hi:
            return a + max(0.0, pm - lo) * (b - a) / (hi - lo)
    return float(bps[-1][3])


def rate(pm, scale="US"):
    """PM2.5 (ug/m3) -> dict(value, cat, color, fg, advice) on the US EPA scale or India's CPCB scale (PM2.5-based). None if missing."""
    if pm is None or pm != pm:
        return None
    india = str(scale).startswith("India")
    v = _interp(pm, IN_BP if india else US_BP)
    cat = next(n for hi, n in (IN_CATS if india else US_CATS) if v <= hi)
    color = (IN_COLORS if india else COLORS)[cat]
    advice = ADVICE["Moderate_IN"] if (india and cat == "Moderate") else ADVICE[cat]
    return {"value": v, "cat": cat, "color": color, "fg": "#1b1b1b" if DARK_TEXT.get(cat) else "#fff", "advice": advice}
