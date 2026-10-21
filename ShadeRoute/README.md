# ShadeRoute 🌳

**A shade-aware walking and community-reporting prototype for the Sime Darby Property University Hackathon.**

ShadeRoute explores greener walking, mobility and climate-resilient township planning. The demo map is centred on an approximate area around Xiamen University Malaysia (XMUM); it is a prototype, not an official Sime Darby Property deployment. 

## Features

- Compare **Shortest**, **More shade** and **Balanced** walking routes.
- Simulate building/tree shadows for a selected date and Malaysia local time, including afternoon while using the app at night.
- Open **Planner** to see the walking network, low-estimated-shade segments and open community reports without calculating a route first.
- Report fallen trees, blocked paths, water spray/slippery paths, crowds, construction and broken pavement.
- Mark reports **Resolved**. Resolved reports stay in history but are hidden from active map pins.
- Calculate **Shortest** with Dijkstra using mapped walking-edge lengths. Unverified inferred links are excluded from route calculation; if the mapped network is disconnected, no route is shown rather than claiming an unverified shortcut is shortest.

## Install (Windows)

Install Anaconda or Miniconda, extract the ZIP, then open Anaconda Prompt in this folder and run:

```bat
conda env create -f environment.yml
conda activate shaderoute_clean
python download_data.py
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`). The data download needs internet access. It downloads walking paths, building footprints and any available mapped trees/green areas from OpenStreetMap.

## Quick demo

1. On **User**, choose Start and click the map; choose Destination and click again.
2. Select a date and time (for example, 3:00 PM), then click **Find routes**.
3. Compare the three options. **Shortest** uses distance as its objective; shade scores do not change its weight.
4. Open **Planner** to view the campus network and reports. Select a date/time to update the shade estimate.
5. To report a problem, select **Report location** on User, click the location, then submit the report below the map. Mark reports resolved when the issue is no longer present. In **Planner → Database management**, you can permanently delete all community reports after confirming.

## Important limitations

Shade values are geometric estimates, not verified measurements or weather forecasts. The Sun position uses the selected timestamp and the map's approximate location. Accuracy depends on mapped building footprints/heights, tree data, canopies, roofs and real site conditions. Some building heights and tree dimensions are assumptions. If no tree features are mapped, the estimate is mainly building-based. A low Sun angle makes shade alternatives unavailable for that selected time. Dashed inferred links are visual hints only and are excluded from route calculation. Dijkstra finds the shortest path between the nearest mapped network nodes, so the result depends on the completeness and correctness of OpenStreetMap walking paths.

Reports are stored in a local SQLite database for this running app instance. To share reports reliably across separate devices or a public deployment, use a hosted database and add user authentication/moderation. OpenStreetMap data requires attribution: https://www.openstreetmap.org/copyright. The prototype uses geometry and solar-position calculations; it does not forecast weather or guarantee shade. 