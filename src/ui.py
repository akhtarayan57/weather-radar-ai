"""HTML building blocks for the phone-app style screens (kept as single-line strings: Markdown treats indented lines as code)."""
import pandas as pd
from . import aqi, wmo

CSS = """<style>
.block-container{max-width:1000px;padding-top:1rem}
header[data-testid=stHeader]{background:transparent}
.topbar{background:#11608f;color:#fff;font-size:22px;font-weight:600;padding:10px 16px;border-radius:12px;margin-bottom:10px;border-bottom:3px solid #f2c200}
.topbar span{color:#f2c200}
.hero{border-radius:16px;color:#fff;padding:16px 18px;margin:8px 0}
.hrow{display:flex;justify-content:space-between;font-size:14px;opacity:.95}
.temp{font-size:84px;font-weight:200;line-height:1.05}.temp .ic{font-size:44px;margin-left:12px}
.cond{font-size:16px;margin-bottom:6px}
.src{font-size:12px;opacity:.92;margin-top:4px}
.chip{display:inline-block;padding:3px 12px;border-radius:6px;font-weight:600;font-size:15px}
.strip{display:flex;gap:8px;overflow-x:auto;padding:8px 2px}
.hc{min-width:78px;background:#fff;border-radius:12px;text-align:center;padding:8px 4px;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.hc .t{font-size:13px;color:#456}.hc .i{font-size:26px}.hc .d{font-size:20px;font-weight:600;color:#c0392b}.hc .p{font-size:12px;color:#2a6da0}
.badge{display:inline-block;min-width:34px;padding:1px 6px;border-radius:5px;font-size:13px;font-weight:700}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0}
.cell{background:#fff;border-radius:12px;padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.12);font-size:14px}
.cell b{display:block;font-size:12px;color:#678;font-weight:500}
.parts{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:8px 0}
.part{background:#fff;border-radius:12px;text-align:center;padding:8px 2px;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.part .n{font-size:12px;color:#567}.part .d{font-size:22px;color:#c0392b}.part .i{font-size:24px}
.row{display:flex;align-items:center;gap:10px;background:#fff;border-radius:10px;padding:8px 12px;margin:4px 0;box-shadow:0 1px 2px rgba(0,0,0,.1);font-size:15px}
.row .dd{width:92px;font-weight:600}.row .tt{flex:1}.row .hi{color:#c0392b;font-weight:600}.row .lo{color:#1f6a96}
.place{border-radius:14px;color:#fff;padding:14px 16px;margin:8px 0 2px;display:flex;justify-content:space-between;align-items:center}
.place .nm{font-size:20px;font-weight:600}.place .tp{font-size:40px;font-weight:200}
h3{margin-top:.6rem}
@media(max-width:600px){.temp{font-size:68px}.parts{grid-template-columns:repeat(2,1fr)}}
</style>"""


def _t12(ts):
    return ts.strftime("%I %p").lstrip("0")


def badge(r):
    """r = aqi.rate(...) dict or None."""
    if not r:
        return ""
    return f'<span class="badge" style="background:{r["color"]};color:{r["fg"]}">{r["value"]:.0f}</span>'


def badge_pm(pm, scale):
    return badge(aqi.rate(pm, scale))


def hero(name, cur, today, chip):
    """chip = {"rate":..., "pm":..., "src":...} or None."""
    desc, icon = wmo.describe(cur["weather_code"], cur["is_day"]); g1, g2 = wmo.sky(cur["weather_code"], cur["is_day"])
    f = lambda s: pd.Timestamp(s).strftime("%I:%M %p").lstrip("0")
    air = ""
    if chip:
        r = chip["rate"]
        air = (f'<span class="chip" style="background:{r["color"]};color:{r["fg"]}">AQI {r["value"]:.0f} · {r["cat"]}</span>'
               f'<div class="src">PM2.5 {chip["pm"]:.0f} µg/m³ · {chip["src"]}</div>')
    return (f'<div class="hero" style="background:linear-gradient(160deg,{g1},{g2})"><div class="hrow"><span>📍 {name}</span>'
            f'<span>🌅 {f(today.sunrise)} · 🌇 {f(today.sunset)}</span></div><div class="temp">{cur["temperature_2m"]:.0f}°'
            f'<span class="ic">{icon}</span></div><div class="cond">{desc} · feels like {cur["apparent_temperature"]:.0f}°</div>{air}</div>')


def hourly_strip(h, pm_h, start, scale, n=24):
    """pm_h = hourly PM2.5 Series (local naive index) or None."""
    out = []
    for t, r in h.loc[start:start + pd.Timedelta(hours=n - 1)].iterrows():
        _, ic = wmo.describe(r.weather_code, r.is_day)
        a = badge_pm(pm_h.get(t), scale) if pm_h is not None else ""
        p = r.precipitation_probability
        out.append(f'<div class="hc"><div class="t">{"Now" if t == start else _t12(t)}</div><div class="i">{ic}</div>'
                   f'<div class="d">{r.temperature_2m:.0f}°</div><div class="p">💧 {0 if p != p else p:.0f}%</div><div>{a}</div></div>')
    return '<div class="strip">' + "".join(out) + "</div>"


def details(cur, hrow, chip):
    ws = wmo.wind_text(cur["wind_speed_10m"], cur["wind_direction_10m"])
    air = "n/a"
    if chip:
        r = chip["rate"]; air = f'{badge(r)} {r["cat"]} · PM2.5 {chip["pm"]:.0f} µg/m³<br><small>{chip["src"]}</small>'
    cells = [("💧 Rain now", f'{cur["precipitation"]:.1f} mm'), ("🌡️ Feels like", f'{cur["apparent_temperature"]:.0f}°'),
             ("🧭 Air pressure", f'{cur["pressure_msl"]:.0f} hPa'), ("💨 Wind", ws),
             ("💦 Relative humidity", f'{cur["relative_humidity_2m"]:.0f} %'), ("🌬️ Gusts up to", f'{cur["wind_gusts_10m"]:.0f} km/h'),
             ("☀️ UV index", f'{hrow.uv_index:.1f}'), ("😷 Air quality", air)]
    return '<div class="grid">' + "".join(f'<div class="cell"><b>{k}</b>{v}</div>' for k, v in cells) + "</div>"


def day_parts(h, pm_h, day, scale):
    out = []
    for label, hr in [("Morning", 9), ("Afternoon", 15), ("Evening", 21), ("Night", 27)]:
        t = day + pd.Timedelta(hours=hr)
        if t not in h.index:
            continue
        r = h.loc[t]; _, ic = wmo.describe(r.weather_code, r.is_day); a = badge_pm(pm_h.get(t), scale) if pm_h is not None else ""
        p = r.precipitation_probability
        out.append(f'<div class="part"><div class="n">{label}</div><div class="d">{r.temperature_2m:.0f}°</div><div class="i">{ic}</div>'
                   f'<div class="p" style="font-size:12px">💧 {0 if p != p else p:.0f}%</div><div>{a}</div></div>')
    return '<div class="parts">' + "".join(out) + "</div>"


def day_rows(d):
    out = []
    for t, r in d.iterrows():
        _, ic = wmo.describe(r.weather_code, 1); p = r.precipitation_probability_max
        out.append(f'<div class="row"><span class="dd">{t:%a %d %b}</span><span>{ic}</span><span class="tt"><span class="hi">{r.temperature_2m_max:.0f}°</span> / '
                   f'<span class="lo">{r.temperature_2m_min:.0f}°</span></span><span>💧 {0 if p != p else p:.0f}%</span><span>🌧️ {r.precipitation_sum:.1f} mm</span><span>☀️ UV {r.uv_index_max:.0f}</span></div>')
    return "".join(out)


def place_card(name, r, air_html=""):
    code = r.get("weather_code", 0); is_day = r.get("is_day", 1)
    is_day = 1 if is_day != is_day else is_day
    g1, g2 = wmo.sky(code, is_day); _, ic = wmo.describe(code, is_day); t = r.get("temperature_2m")
    return (f'<div class="place" style="background:linear-gradient(160deg,{g1},{g2})"><div><div class="nm">{name}</div>'
            f'<div>{ic} {air_html}</div></div><div class="tp">{"n/a" if t is None or t != t else f"{t:.0f}°"}</div></div>')
