"""WMO weather codes (Open-Meteo) -> text, emoji and sky colours."""
CODES = {0: ("Clear sky", "☀️", "🌙"), 1: ("Mainly clear", "🌤️", "🌙"), 2: ("Partly cloudy", "⛅", "☁️"), 3: ("Overcast", "☁️", "☁️"),
         45: ("Fog", "🌫️", "🌫️"), 48: ("Rime fog", "🌫️", "🌫️"), 51: ("Light drizzle", "🌦️", "🌧️"), 53: ("Drizzle", "🌦️", "🌧️"),
         55: ("Heavy drizzle", "🌧️", "🌧️"), 56: ("Freezing drizzle", "🌧️", "🌧️"), 57: ("Freezing drizzle", "🌧️", "🌧️"),
         61: ("Light rain", "🌦️", "🌧️"), 63: ("Rain", "🌧️", "🌧️"), 65: ("Heavy rain", "🌧️", "🌧️"), 66: ("Freezing rain", "🌧️", "🌧️"),
         67: ("Freezing rain", "🌧️", "🌧️"), 71: ("Light snow", "🌨️", "🌨️"), 73: ("Snow", "🌨️", "🌨️"), 75: ("Heavy snow", "❄️", "❄️"),
         77: ("Snow grains", "🌨️", "🌨️"), 80: ("Rain showers", "🌦️", "🌧️"), 81: ("Rain showers", "🌧️", "🌧️"),
         82: ("Violent showers", "⛈️", "⛈️"), 85: ("Snow showers", "🌨️", "🌨️"), 86: ("Snow showers", "🌨️", "🌨️"),
         95: ("Thunderstorm", "⛈️", "⛈️"), 96: ("Thunderstorm, hail", "⛈️", "⛈️"), 99: ("Thunderstorm, hail", "⛈️", "⛈️")}


def _c(code):
    try:
        return int(code)
    except (TypeError, ValueError):
        return -1


def describe(code, is_day=1):
    name, d, n = CODES.get(_c(code), ("Unknown", "❔", "❔"))
    return name, (d if is_day else n)


def sky(code, is_day=1):
    c = _c(code)
    if not is_day:
        return ("#0f1b33", "#24406b") if c in (0, 1, 2) else ("#1a2536", "#3a4a5f")
    if c in (0, 1):
        return "#2b7fd4", "#74b9ff"
    if c in (2, 3):
        return "#5f7d95", "#9db4c5"
    if c in (45, 48):
        return "#7d8b98", "#b5bfc8"
    if c in (71, 73, 75, 77, 85, 86):
        return "#8aa4bd", "#cfdbe6"
    if c in (95, 96, 99, 82):
        return "#2c3a4d", "#59687a"
    return "#3d6a8c", "#7fa5c0"


def wind_text(speed, deg):
    s = "light winds" if speed < 12 else "moderate winds" if speed < 29 else "strong winds" if speed < 50 else "storm-force winds"
    d = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((deg or 0) + 22.5) // 45) % 8]
    return f"{s} from {d} ({speed:.0f} km/h)"
