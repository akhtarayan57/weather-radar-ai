"""Build the global forecast field (6 degree grid, 72 h) shown on the World map. No API key needed. Takes about 10 minutes
(it is paced to respect Open-Meteo's free limits). Run twice a day by .github/workflows/world_field.yml."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import fieldmap

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "data" / "live" / "world_field.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = fieldmap.fetch_field("World (fine)", True, hours=72, pause=9)
    fieldmap.save_field(d, out)
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB), layers: {list(d['layers'])}, frames: {len(d['times'])}")


if __name__ == "__main__":
    main()
