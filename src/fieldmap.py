"""Browser maps (Leaflet, free keyless tiles): colour-filled field + wind particles, and a rain-radar view.
Original doc: Animated 'Windy-style' map: colour-filled field + moving wind particles, on a Leaflet map.
Data = regular lat/lon grid of Open-Meteo forecasts (free, no key). Everything after loading runs in the browser."""
import json
import time
import numpy as np
import pandas as pd
from .common import get_json
from .world import AQ, FC

# name: (lat_min, lat_max, lon_min, lon_max, step_deg)  ~300-450 points each, to respect free-API rate limits
REGIONS = {
    "India": (6, 38, 66, 100, 1.5), "World": (-60, 72, -180, 180, 12), "Europe": (34, 72, -12, 42, 2.5),
    "East & SE Asia": (-10, 55, 90, 150, 3.5), "North America": (15, 70, -170, -50, 4),
    "Africa & Middle East": (-35, 40, -20, 60, 4), "South America": (-56, 14, -82, -34, 3.5),
    "Australia & Oceania": (-48, 0, 110, 180, 3.5)}
FINE_WORLD = (-60, 78, -180, 180, 6)  # 1464 points; built by scripts/build_world_field.py (not fetched live)
HOURLY_W = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation",
            "cloud_cover", "pressure_msl"]
HOURLY_A = ["pm2_5", "us_aqi"]
EPA = {"pm2_5": [[0, "#00e400"], [12, "#ffff00"], [35, "#ff7e00"], [55, "#ff0000"], [125, "#8f3f97"], [250, "#7e0023"]],
       "us_aqi": [[0, "#00e400"], [50, "#ffff00"], [100, "#ff7e00"], [150, "#ff0000"], [200, "#8f3f97"], [300, "#7e0023"]]}
META = {  # key: (label, unit, colour stops, optional fade range [transparent below a, opaque above b])
    "temperature_2m": ("Temperature", "°C", [[-30, "#5b2a86"], [-15, "#3a4fb5"], [0, "#3fa0e0"], [10, "#4cc9a0"], [20, "#f2e35a"], [30, "#f5a04a"], [40, "#d6342f"], [50, "#7a1020"]], None),
    "wind_speed_10m": ("Wind speed", "km/h", [[0, "#2a3b6e"], [10, "#3fa0e0"], [20, "#4cc97a"], [35, "#f2e35a"], [55, "#f5a04a"], [80, "#d6342f"], [110, "#7a1020"]], None),
    "relative_humidity_2m": ("Humidity", "%", [[0, "#8c6d3f"], [30, "#d9c27a"], [60, "#5fb6a6"], [100, "#1f4fa3"]], None),
    "cloud_cover": ("Cloud cover", "%", [[0, "#9fb3c8"], [50, "#c3d0dc"], [100, "#ffffff"]], [5, 50]),
    "precipitation": ("Precipitation", "mm/h", [[0, "#2f6fba"], [1, "#38b6e0"], [3, "#7ee08a"], [8, "#f2e35a"], [20, "#d6342f"]], [0.05, 0.6]),
    "pressure_msl": ("Pressure (sea level)", "hPa", [[980, "#5b2a86"], [1000, "#3a6fd0"], [1013, "#4cc9a0"], [1025, "#f2e35a"], [1040, "#d6342f"]], None),
    "pm2_5": ("PM2.5 (CAMS model)", "µg/m³", EPA["pm2_5"], None),
    "us_aqi": ("US AQI (CAMS model)", "", EPA["us_aqi"], None)}


def grid_axes(region):
    la0, la1, lo0, lo1, st = FINE_WORLD if region == "World (fine)" else REGIONS[region]
    return np.round(np.arange(la0, la1 + 1e-6, st), 2), np.round(np.arange(lo0, lo1 + 1e-6, st), 2)


def _fetch(url, pts, cols, chunk=50, pause=0.4, days=3):
    out = []
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        j = get_json(url, {"latitude": ",".join(f"{x[0]:.2f}" for x in p), "longitude": ",".join(f"{x[1]:.2f}" for x in p),
                           "hourly": ",".join(cols), "forecast_days": days, "timezone": "UTC"})
        out += j if isinstance(j, list) else [j]
        if pause > 2:
            print(f"  {len(out)}/{len(pts)} points", flush=True)
        time.sleep(pause)
    return out


def fetch_field(region, with_air=True, step_h=3, hours=48, pause=0.4):
    """Forecast fields on a grid. Returns a JSON-ready dict (row 0 = southernmost latitude)."""
    lats, lons = grid_axes(region)
    pts = [(float(a), float(b)) for a in lats for b in lons]
    days = 3 if hours <= 48 else 4
    w = _fetch(FC, pts, HOURLY_W, pause=pause, days=days)
    t = pd.to_datetime(w[0]["hourly"]["time"], utc=True)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    idx = [i for i, x in enumerate(t) if x >= now][:hours + 1:step_h]

    def arr(src, var):
        a = np.array([[(r["hourly"][var][i] if r["hourly"][var][i] is not None else np.nan) for i in idx] for r in src], float).T
        for k in range(a.shape[0]):
            m = np.nanmean(a[k]) if np.isfinite(a[k]).any() else 0.0
            a[k] = np.where(np.isfinite(a[k]), a[k], m)
        return a  # (T, npts)

    fields = {v: arr(w, v) for v in HOURLY_W if v != "wind_direction_10m"}
    ms, deg = fields["wind_speed_10m"] / 3.6, np.deg2rad(arr(w, "wind_direction_10m"))
    u, v = -ms * np.sin(deg), -ms * np.cos(deg)  # direction is where wind comes FROM
    if with_air:
        try:
            a = _fetch(AQ, pts, HOURLY_A, pause=pause, days=days)
            ta = pd.to_datetime(a[0]["hourly"]["time"], utc=True)
            ia = [int(np.argmin(np.abs((ta - t[i]).total_seconds()))) for i in idx]
            for var in HOURLY_A:
                x = np.array([[(r["hourly"][var][i] if r["hourly"][var][i] is not None else np.nan) for i in ia] for r in a], float).T
                for k in range(x.shape[0]):
                    x[k] = np.where(np.isfinite(x[k]), x[k], np.nanmean(x[k]) if np.isfinite(x[k]).any() else 0.0)
                fields[var] = x
        except Exception:  # air-quality API unavailable: still show weather
            pass
    rnd = lambda m, d=1: [[round(float(z), d) for z in row] for row in m]
    layers = {k: {"label": META[k][0], "unit": META[k][1], "stops": META[k][2], "fade": META[k][3], "frames": rnd(fields[k])}
              for k in META if k in fields}
    return {"region": region, "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "lats": [float(x) for x in lats], "lons": [float(x) for x in lons],
            "times": [t[i].strftime("%Y-%m-%dT%H:%M:%SZ") for i in idx], "layers": layers, "u": rnd(u, 2), "v": rnd(v, 2)}



def save_field(data, path):
    import gzip
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def load_field(path, min_frames=6):
    """Load a saved field and drop frames older than 3 h. Returns None if missing or too old to be useful."""
    import gzip, os
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)
    keep = [i for i, t in enumerate(d["times"]) if pd.Timestamp(t) >= cutoff]
    if len(keep) < min_frames:
        return None
    d["times"] = [d["times"][i] for i in keep]
    d["u"], d["v"] = [d["u"][i] for i in keep], [d["v"][i] for i in keep]
    for L in d["layers"].values():
        L["frames"] = [L["frames"][i] for i in keep]
    return d


def region_for(lat, lon):
    """Smallest preset region containing the point (World if none)."""
    best = None
    for n, (la0, la1, lo0, lo1, _) in REGIONS.items():
        if n != "World" and la0 <= lat <= la1 and lo0 <= lon <= lo1:
            area = (la1 - la0) * (lo1 - lo0)
            if best is None or area < best[0]:
                best = (area, n)
    return best[1] if best else "World"


def prepare(data, keep=None, wind=True, home=None, start=None):
    """Copy of the cached field data, optionally restricted to some layers, with a 'home' marker."""
    d = dict(data)
    if keep:
        d["layers"] = {k: v for k, v in data["layers"].items() if k in keep} or data["layers"]
    d["wind"], d["home"], d["start"] = wind, home, start
    if not wind:
        d["u"], d["v"] = [], []
    return d


BASE_JS = r"""
function addBase(map){
 map.createPane('labels');map.getPane('labels').style.zIndex=350;map.getPane('labels').style.pointerEvents='none';
 const E='https://server.arcgisonline.com/ArcGIS/rest/services/';let errs=0,fell=false;
 const esri=L.tileLayer(E+'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',{attribution:'Tiles © Esri',maxNativeZoom:16,maxZoom:16,noWrap:true}).addTo(map);
 const lab=L.tileLayer(E+'Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:16,maxZoom:16,pane:'labels',noWrap:true}).addTo(map);
 esri.on('tileerror',()=>{if(fell||++errs<4)return;fell=true;map.removeLayer(esri);map.removeLayer(lab);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors',maxZoom:19,noWrap:true}).addTo(map);});
}
function addBorders(map,color){
 const urls=['https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json','https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson'];
 (function go(i){if(i>=urls.length)return;fetch(urls[i]).then(r=>r.json()).then(g=>{
  L.geoJSON(g,{pane:'labels',interactive:false,style:{color:color||'#40566b',weight:0.9,fill:false,opacity:0.85}}).addTo(map);}).catch(()=>go(i+1));})(0);
}
function fitTo(map,b){map.invalidateSize();const bb=L.latLngBounds(b);let z=map.getBoundsZoom(bb,false);z=Math.ceil(z*4)/4;map.setView(bb.getCenter(),z,{animate:false});}
"""

LIGHT_CSS = """
html,body{margin:0;height:100%;background:#eaf1f7;font-family:system-ui,Segoe UI,sans-serif}
#map{position:absolute;inset:0}
.panel{position:absolute;z-index:1000;background:rgba(255,255,255,.95);color:#18324a;padding:8px 10px;border-radius:8px;font-size:13px;box-shadow:0 1px 6px rgba(0,0,0,.3)}
select,button{background:#fff;color:#18324a;border:1px solid #9bb;border-radius:6px;padding:4px 8px}
input[type=range]{flex:1}
#bot{left:10px;right:10px;bottom:30px;display:flex;gap:10px;align-items:center}
"""


def satellite_html(lat, lon, name, date):
    return SAT.replace("__BASE__", BASE_JS).replace("__CSS__", LIGHT_CSS).replace("__HOME__", json.dumps([lat, lon, name])).replace("__DATE__", date)


def build_html(data):
    return TEMPLATE.replace("__BASE__", BASE_JS).replace("__CSS__", LIGHT_CSS).replace("__DATA__", json.dumps(data, separators=(",", ":")))


def radar_html(lat, lon, name):
    return RADAR.replace("__BASE__", BASE_JS).replace("__CSS__", LIGHT_CSS).replace("__HOME__", json.dumps([lat, lon, name]))


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>__CSS__
#top{top:10px;left:55px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;max-width:calc(100% - 290px)}
#leg{top:10px;right:10px;width:210px}
#bar{height:12px;border-radius:6px;margin:4px 0}
#lab{display:flex;justify-content:space-between;font-size:11px;opacity:.85}
.leaflet-popup-content{font-size:13px;line-height:1.5}
@media(max-width:600px){#leg{top:auto;bottom:80px;width:170px}#top{max-width:calc(100% - 65px)}}
</style></head><body><div id="map"></div>
<div id="top" class="panel"><b>Layer</b><select id="layer"></select><label>Opacity <input type="range" id="op" min="20" max="90" value="50" style="width:70px;flex:none"></label><label id="wl"><input type="checkbox" id="wind" checked> Wind animation</label></div>
<div id="leg" class="panel"><div id="lt"></div><div id="bar"></div><div id="lab"><span id="l0"></span><span id="l1"></span></div></div>
<div id="bot" class="panel"><button id="play">▶</button><input id="t" type="range" min="0" value="0"><span id="tl"></span></div>
<script>
const D=__DATA__;__BASE__
const ny=D.lats.length,nx=D.lons.length,la0=D.lats[0],la1=D.lats[ny-1],lo0=D.lons[0],lo1=D.lons[nx-1];
const dla=(la1-la0)/(ny-1),dlo=(lo1-lo0)/(nx-1),T=D.times.length;
let cur=(D.start&&D.layers[D.start])?D.start:(D.layers.us_aqi?'us_aqi':Object.keys(D.layers)[0]),fi=0,ALPHA=0.5,windOn=D.wind!==false,moving=false,timer=null;
if(D.wind===false){document.getElementById('wl').style.display='none';}
function bil(a,lat,lon){const fy=(lat-la0)/dla,fx=(lon-lo0)/dlo;if(!(fy>=0&&fy<=ny-1&&fx>=0&&fx<=nx-1))return null;
 const y0=Math.min(Math.floor(fy),ny-2),x0=Math.min(Math.floor(fx),nx-2),ty=fy-y0,tx=fx-x0;
 return (a[y0*nx+x0]*(1-tx)+a[y0*nx+x0+1]*tx)*(1-ty)+(a[(y0+1)*nx+x0]*(1-tx)+a[(y0+1)*nx+x0+1]*tx)*ty;}
function cr(p0,p1,p2,p3,t){return 0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t);}
function bic(a,lat,lon){const fy=(lat-la0)/dla,fx=(lon-lo0)/dlo;if(!(fy>=0&&fy<=ny-1&&fx>=0&&fx<=nx-1))return null;
 const y0=Math.floor(fy),x0=Math.floor(fx),ty=fy-y0,tx=fx-x0,r=[];
 for(let j=-1;j<=2;j++){const yy=Math.min(ny-1,Math.max(0,y0+j)),g=i=>a[yy*nx+Math.min(nx-1,Math.max(0,x0+i))];r.push(cr(g(-1),g(0),g(1),g(2),tx));}
 return cr(r[0],r[1],r[2],r[3],ty);}
function hex(s){return [parseInt(s.slice(1,3),16),parseInt(s.slice(3,5),16),parseInt(s.slice(5,7),16)];}
function buildLut(L){const st=L.stops,rgb=st.map(s=>hex(s[1])),N=512,out=new Uint8ClampedArray(N*3);L.vmin=st[0][0];L.vmax=st[st.length-1][0];
 for(let i=0;i<N;i++){const v=L.vmin+(L.vmax-L.vmin)*i/(N-1);let k=0;while(k<st.length-2&&v>st[k+1][0])k++;
  const t=Math.min(1,Math.max(0,(v-st[k][0])/(st[k+1][0]-st[k][0])));for(let c=0;c<3;c++)out[i*3+c]=rgb[k][c]+(rgb[k+1][c]-rgb[k][c])*t;}
 L.lut=out;}
Object.values(D.layers).forEach(buildLut);
const map=L.map('map',{zoomAnimation:false,zoomSnap:0.25,minZoom:1,maxBounds:[[-85,-180],[85,180]],maxBoundsViscosity:1});
addBase(map);addBorders(map);
function fit(){fitTo(map,[[la0,lo0],[la1,lo1]]);}
fit();setTimeout(fit,300);setTimeout(fit,1200);
if(D.home){L.circleMarker([D.home[0],D.home[1]],{radius:7,color:'#fff',weight:2,fillColor:'#d6342f',fillOpacity:1}).addTo(map).bindTooltip(D.home[2]);}
const pane=map.createPane('fld');pane.style.zIndex=300;pane.style.pointerEvents='none';
const fc=L.DomUtil.create('canvas','',pane),pcv=L.DomUtil.create('canvas','',pane),pctx=pcv.getContext('2d');
function place(){const p=map.containerPointToLayerPoint([0,0]);L.DomUtil.setPosition(fc,p);L.DomUtil.setPosition(pcv,p);}
function drawField(step){const S=map.getSize(),W=S.x,H=S.y;fc.width=W;fc.height=H;place();
 const Ly=D.layers[cur],a=Ly.frames[fi],cw=Math.ceil(W/step),ch=Math.ceil(H/step),tmp=document.createElement('canvas');tmp.width=cw;tmp.height=ch;
 const tc=tmp.getContext('2d'),img=tc.createImageData(cw,ch);
 for(let j=0;j<ch;j++)for(let i=0;i<cw;i++){const ll=map.containerPointToLatLng([i*step+step/2,j*step+step/2]),v=bic(a,ll.lat,ll.lng),o=(j*cw+i)*4;
  if(v===null){img.data[o+3]=0;continue;}
  const t=Math.min(1,Math.max(0,(v-Ly.vmin)/(Ly.vmax-Ly.vmin))),k=Math.round(t*511)*3;
  img.data[o]=Ly.lut[k];img.data[o+1]=Ly.lut[k+1];img.data[o+2]=Ly.lut[k+2];
  let al=ALPHA;if(Ly.fade)al*=Math.min(1,Math.max(0,(v-Ly.fade[0])/(Ly.fade[1]-Ly.fade[0])));img.data[o+3]=al*255;}
 tc.putImageData(img,0,0);const c=fc.getContext('2d');c.imageSmoothingEnabled=true;c.imageSmoothingQuality='high';c.clearRect(0,0,W,H);c.drawImage(tmp,0,0,cw*step,ch*step);}
let parts=[];const K=0.3,COL=['rgba(30,60,110,.55)','rgba(15,100,70,.6)','rgba(120,90,0,.65)','rgba(170,70,0,.7)','rgba(150,20,20,.75)'];
function seed(){const S=map.getSize();pcv.width=S.x;pcv.height=S.y;place();const n=windOn?Math.min(4500,Math.max(1200,Math.round(S.x*S.y/260))):0;
 parts=Array.from({length:n},()=>({x:Math.random()*S.x,y:Math.random()*S.y,age:Math.floor(Math.random()*90)}));}
function loop(){requestAnimationFrame(loop);if(moving||!windOn)return;const S=map.getSize();
 pctx.globalCompositeOperation='destination-out';pctx.fillStyle='rgba(0,0,0,0.07)';pctx.fillRect(0,0,S.x,S.y);
 pctx.globalCompositeOperation='source-over';pctx.lineWidth=1.3;const paths=[[],[],[],[],[]];
 for(const p of parts){const ll=map.containerPointToLatLng([p.x,p.y]),u=bil(D.u[fi],ll.lat,ll.lng);
  if(u===null||p.age>100||p.x<0||p.y<0||p.x>S.x||p.y>S.y){p.x=Math.random()*S.x;p.y=Math.random()*S.y;p.age=0;continue;}
  const v=bil(D.v[fi],ll.lat,ll.lng),sp=Math.hypot(u,v),x2=p.x+u*K,y2=p.y-v*K;
  paths[Math.min(4,Math.floor(sp/3))].push(p.x,p.y,x2,y2);p.x=x2;p.y=y2;p.age++;}
 for(let b=0;b<5;b++){const a=paths[b];if(!a.length)continue;pctx.strokeStyle=COL[b];pctx.beginPath();for(let i=0;i<a.length;i+=4){pctx.moveTo(a[i],a[i+1]);pctx.lineTo(a[i+2],a[i+3]);}pctx.stroke();}}
function legend(){const Ly=D.layers[cur],st=Ly.stops,r=Ly.vmax-Ly.vmin;
 document.getElementById('lt').textContent=Ly.label+(Ly.unit?' ('+Ly.unit+')':'');
 document.getElementById('bar').style.background='linear-gradient(to right,'+st.map(s=>s[1]+' '+((s[0]-Ly.vmin)/r*100).toFixed(1)+'%').join(',')+')';
 document.getElementById('l0').textContent=Ly.vmin;document.getElementById('l1').textContent=Ly.vmax;}
function tlabel(){document.getElementById('tl').textContent=new Date(D.times[fi]).toUTCString().slice(0,22)+' UTC';document.getElementById('t').value=fi;}
const sel=document.getElementById('layer');Object.keys(D.layers).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=D.layers[k].label;sel.appendChild(o);});sel.value=cur;
sel.onchange=()=>{cur=sel.value;legend();drawField(4);};
document.getElementById('op').oninput=e=>{ALPHA=e.target.value/100;drawField(4);};
document.getElementById('wind').onchange=e=>{windOn=e.target.checked;if(!windOn)pctx.clearRect(0,0,pcv.width,pcv.height);else seed();};
const sl=document.getElementById('t');sl.max=T-1;sl.oninput=()=>{fi=+sl.value;tlabel();drawField(4);};
document.getElementById('play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='▶';}else{this.textContent='❚❚';timer=setInterval(()=>{fi=(fi+1)%T;tlabel();drawField(4);},1000);}};
function refresh(){moving=false;pcv.style.visibility='visible';drawField(4);seed();}
map.on('movestart',()=>{moving=true;pcv.style.visibility='hidden';});map.on('moveend',refresh);map.on('resize',refresh);
const ADV={'Good':'Air is clean. Enjoy outdoor activities.','Moderate':'Unusually sensitive people should reduce long, heavy outdoor exertion.','Unhealthy for sensitive groups':'Children, older adults and people with heart or lung disease should limit prolonged outdoor exertion.','Unhealthy':'Everyone should reduce prolonged outdoor exertion; sensitive groups stay indoors.','Very unhealthy':'Avoid outdoor exertion; sensitive groups stay indoors.','Hazardous':'Stay indoors, windows closed; avoid outdoor activity.'};
const CC={'Good':'#00e400','Moderate':'#ffff00','Unhealthy for sensitive groups':'#ff7e00','Unhealthy':'#ff0000','Very unhealthy':'#8f3f97','Hazardous':'#7e0023'};
function catPm(v){return v<=9?'Good':v<=35.4?'Moderate':v<=55.4?'Unhealthy for sensitive groups':v<=125.4?'Unhealthy':v<=225.4?'Very unhealthy':'Hazardous';}
function catAqi(v){return v<=50?'Good':v<=100?'Moderate':v<=150?'Unhealthy for sensitive groups':v<=200?'Unhealthy':v<=300?'Very unhealthy':'Hazardous';}
map.on('click',e=>{const {lat,lng}=e.latlng;let h='<b>'+lat.toFixed(2)+'°, '+lng.toFixed(2)+'°</b><br>',any=false,c=null;
 for(const k in D.layers){const Ly=D.layers[k],v=bil(Ly.frames[fi],lat,lng);if(v===null)continue;any=true;
  if(k==='us_aqi')c=catAqi(v);else if(k==='pm2_5'&&!c)c=catPm(v);
  h+=Ly.label+': <b>'+v.toFixed(1)+'</b> '+Ly.unit+'<br>';}
 if(c)h+='<span style="color:'+CC[c]+'">●</span> <b>'+c+'</b><br>'+ADV[c]+'<br>';
 if(D.wind!==false){const u=bil(D.u[fi],lat,lng),vv=bil(D.v[fi],lat,lng);if(u!==null)h+='Wind from '+Math.round((Math.atan2(-u,-vv)*180/Math.PI+360)%360)+'°<br>';}
 h+='<small>Forecast grid values, '+document.getElementById('tl').textContent+'</small>';
 if(any)L.popup().setLatLng(e.latlng).setContent(h).openOn(map);});
legend();tlabel();setTimeout(()=>{refresh();loop();},250);
</script></body></html>"""

RADAR = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>__CSS__
#info{top:10px;left:55px;max-width:calc(100% - 70px);font-size:12px}
</style></head><body><div id="map"></div>
<div id="info" class="panel">🌧️ Rain radar (past 2 h). Radar covers only some regions; free data stops at zoom level 7.</div>
<div id="bot" class="panel"><button id="play">▶</button><input id="t" type="range" min="0" max="0" value="0"><span id="tl">loading radar…</span></div>
<script>
const HOME=__HOME__;__BASE__
const map=L.map('map',{minZoom:2,maxZoom:12}).setView([HOME[0],HOME[1]],6);addBase(map);
map.attributionControl.addAttribution('Radar © <a href="https://www.rainviewer.com/">RainViewer</a>');
L.circleMarker([HOME[0],HOME[1]],{radius:7,color:'#fff',weight:2,fillColor:'#d6342f',fillOpacity:1}).addTo(map).bindTooltip(HOME[2]);
map.createPane('rad');map.getPane('rad').style.zIndex=300;map.getPane('rad').style.pointerEvents='none';
let frames=[],layers=[],idx=0,timer=null,host='',busy=false;
const tl=document.getElementById('tl'),sl=document.getElementById('t'),btn=document.getElementById('play');
function url(i){return host+frames[i].path+'/256/{z}/{x}/{y}/2/1_1.png';}
function ensure(i){if(!layers[i])layers[i]=L.tileLayer(url(i),{opacity:0,maxNativeZoom:7,maxZoom:12,pane:'rad'}).addTo(map);return layers[i];}
function label(){const t=new Date(frames[idx].time*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});tl.textContent=t+(idx===frames.length-1?' (latest)':'')+' · your device time';}
function show(i){idx=i;layers.forEach((l,k)=>{if(l)l.setOpacity(k===i?0.75:0);});ensure(i).setOpacity(0.75);sl.value=i;label();}
fetch('https://api.rainviewer.com/public/weather-maps.json').then(r=>r.json()).then(j=>{
 host=j.host||'https://tilecache.rainviewer.com';const past=(j.radar&&j.radar.past)||[];
 frames=past.filter((f,i)=>(past.length-1-i)%2===0);if(!frames.length){tl.textContent='Radar data unavailable right now';return;}
 sl.max=frames.length-1;show(frames.length-1);}).catch(()=>{tl.textContent='Radar data unavailable right now';});
sl.oninput=()=>{if(frames.length)show(+sl.value);};
btn.onclick=function(){if(!frames.length)return;if(timer||busy){clearInterval(timer);timer=null;busy=false;this.textContent='▶';return;}
 this.textContent='❚❚';busy=true;let k=0;
 (function pre(){if(!busy)return;if(k<frames.length){tl.textContent='Loading frames '+(k+1)+'/'+frames.length+'…';ensure(k);k++;setTimeout(pre,2500);}
  else{busy=false;timer=setInterval(()=>show((idx+1)%frames.length),800);}})();};
</script></body></html>"""


SAT = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>__CSS__
#top{top:10px;left:55px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;max-width:calc(100% - 70px)}
#msg{left:10px;bottom:30px;max-width:calc(100% - 20px);font-size:12px}
</style></head><body><div id="map"></div>
<div id="top" class="panel"><label><input type="checkbox" id="tc" checked> Satellite photo</label>
<label>Overlay <select id="ov"><option value="aod">Haze, smoke, dust (aerosols)</option><option value="fire">Active fires</option><option value="none">None</option></select></label></div>
<div id="msg" class="panel">NASA GIBS satellite images for __DATE__ (UTC). Aerosol haze is not the same as ground-level PM2.5. Gaps are clouds or missed satellite passes.</div>
<script>
const HOME=__HOME__,DATE='__DATE__';__BASE__
const map=L.map('map',{minZoom:1,maxZoom:12,zoomSnap:0.25,maxBounds:[[-85,-180],[85,180]],maxBoundsViscosity:1}).setView([HOME[0],HOME[1]],4);
map.createPane('labels');map.getPane('labels').style.zIndex=350;map.getPane('labels').style.pointerEvents='none';
map.createPane('sat');map.getPane('sat').style.zIndex=250;map.createPane('ovl');map.getPane('ovl').style.zIndex=300;
const G='https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/';
function gibs(layer,tms,ext,nz,pane,op){let errs=0;const l=L.tileLayer(G+layer+'/default/'+DATE+'/'+tms+'/{z}/{y}/{x}.'+ext,{maxNativeZoom:nz,maxZoom:12,noWrap:true,pane:pane,opacity:op,
  bounds:[[-85.05,-179.99],[85.05,179.99]],attribution:'<a href="https://earthdata.nasa.gov/gibs">NASA EOSDIS GIBS</a>'});
 l.on('tileerror',()=>{if(++errs===6)document.getElementById('msg').textContent='Some satellite tiles are not available for '+DATE+' (try an earlier date).';});return l;}
const esri='https://server.arcgisonline.com/ArcGIS/rest/services/';
L.tileLayer(esri+'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:16,maxZoom:16,noWrap:true,attribution:'Tiles © Esri'}).addTo(map);
const photo=gibs('VIIRS_SNPP_CorrectedReflectance_TrueColor','GoogleMapsCompatible_Level9','jpg',9,'sat',1).addTo(map);
const LAY={aod:gibs('MODIS_Combined_Value_Added_AOD','GoogleMapsCompatible_Level6','png',6,'ovl',0.8),fire:gibs('VIIRS_SNPP_Thermal_Anomalies_375m_All','GoogleMapsCompatible_Level9','png',9,'ovl',1)};
let cur='aod';LAY.aod.addTo(map);
L.tileLayer(esri+'Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:13,maxZoom:16,pane:'labels',noWrap:true}).addTo(map);
L.circleMarker([HOME[0],HOME[1]],{radius:7,color:'#fff',weight:2,fillColor:'#d6342f',fillOpacity:1}).addTo(map).bindTooltip(HOME[2]);
document.getElementById('tc').onchange=e=>{if(e.target.checked)photo.addTo(map);else map.removeLayer(photo);};
document.getElementById('ov').onchange=e=>{if(LAY[cur])map.removeLayer(LAY[cur]);cur=e.target.value;if(LAY[cur])LAY[cur].addTo(map);};
setTimeout(()=>map.invalidateSize(),300);
</script></body></html>"""
