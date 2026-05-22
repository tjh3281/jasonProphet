"""
Rewatt x Prophet Demo
=====================
Forecasts tomorrow's hourly occupancy for "Lecture Hall B" from four weeks of
synthetic timetable history, then derives the optimal AC-shutdown trigger.

Prophet understands day-of-week patterns natively, so Monday morning peaks,
Friday light loads, and empty weekends are all learnt automatically.

HOW TO RUN (VS Code)
--------------------
1. Open a terminal in VS Code (Ctrl+`) and create a virtual environment:
     python -m venv .venv
     .venv\Scripts\activate          # Windows
     source .venv/bin/activate       # macOS / Linux
2. Install dependencies:
     pip install -r requirements.txt
3. Open this file in VS Code. Each `# %%` block is a cell.
   Click "Run Cell" above each block, or hit Shift+Enter to run cell-by-cell.

WHAT YOU GET
------------
- history.png  : four weeks of synthetic lecture-hall occupancy
- forecast.png : history + Prophet forecast + shutdown marker
- Printed line you can read out during the pitch:
    "ACTION: Schedule AC shutdown at HH:MM on <Weekday>"
"""

# %% [markdown]
# ## 1. Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet

np.random.seed(42)
plt.rcParams["figure.dpi"] = 110


# %% [markdown]
# ## 2. Generate four weeks of synthetic timetable data
#
# Lecture Hall B, capacity 120. A weekday-specific timetable (a real
# WEEKLY pattern). DAYS=28 means the loop walks four calendar weeks, so
# weeks 2-4 automatically replay week 1's schedule with fresh small
# random noise. Weekends are empty; that's the AC-shutdown opportunity.

# %%
START = datetime(2026, 4, 22, 0, 0)   # a Wednesday -> 28 days later
                                      # history ends Tue, so the
                                      # forecast day lands on a WEDNESDAY
DAYS  = 28
HOURS = DAYS * 24
CAPACITY = 120

# (weekday, hour_start, hour_end, occupancy_fraction)
# weekday: 0=Mon ... 4=Fri. SAT/SUN (5/6) have no entries -> completely empty.
TIMETABLE = [
    (0,  8, 10, 0.85),  # Mon 08-10  WIA1003
    (0, 14, 16, 0.70),  # Mon 14-16  WIX1002
    (1,  9, 11, 0.90),  # Tue 09-11  WIA1001
    (1, 13, 15, 0.60),  # Tue 13-15  WIX1005
    (2,  8, 10, 0.85),  # Wed 08-10  WIA1003
    (2, 11, 12, 0.40),  # Wed 11-12  Seminar
    (3,  9, 11, 0.90),  # Thu 09-11  Physics
    (3, 14, 17, 0.75),  # Thu 14-17  Engineering Studio
    (4, 10, 12, 0.65),  # Fri 10-12  Elective  <- only ONE class on Friday
    # (no Sat/Sun entries -> weekends totally empty)
]

records = []
for h in range(HOURS):
    ts = START + timedelta(hours=h)
    wd, hh = ts.weekday(), ts.hour

    occ = 0.0
    for d, hs, he, frac in TIMETABLE:
        if wd == d and hs <= hh < he:
            target = CAPACITY * frac
            noise  = np.random.normal(0, 1)
            occ = max(0.0, target + noise)
            break

    # Ambient: cleaners / passers-by during weekday daytime
    if wd < 5 and 7 <= hh <= 18 and occ == 0:
        occ = max(0.0, np.random.uniform(0, 2.5))

    records.append({"timestamp": ts, "occupancy": occ})

df = pd.DataFrame(records)
print(df.head(12).to_string(index=False))
print(f"\nTotal hours of history: {len(df)}")


# %% [markdown]
# ## 3. Visualise the history (sanity check)

# %%
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.plot(df["timestamp"], df["occupancy"], lw=0.8, color="#2c3e50")
ax.set_title("Lecture Hall B — four weeks of synthetic hourly occupancy",
             weight="bold")
ax.set_ylabel("People present")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("history.png")
plt.show()


# %% [markdown]
# ## 4. Fit Prophet model
#
# Prophet requires columns named `ds` (datetime) and `y` (value).
# weekly_seasonality and daily_seasonality let it learn day-of-week
# and hour-of-day patterns from the four weeks of history.

# %%
prophet_df = df.rename(columns={"timestamp": "ds", "occupancy": "y"})

model = Prophet(
    weekly_seasonality=True,   # learns Mon-Sun pattern
    daily_seasonality=True,    # learns hour-of-day pattern
    seasonality_mode="additive",
    interval_width=0.80,       # 80% confidence band
)
model.add_country_holidays(country_name="MY")  # Malaysia public holidays
model.fit(prophet_df)
print("Prophet model fitted.")


# %% [markdown]
# ## 5. Forecast tomorrow's 24 hours

# %%
last_ts = df["timestamp"].iloc[-1]
future_ts = [last_ts + timedelta(hours=i + 1) for i in range(24)]

future_df = pd.DataFrame({"ds": future_ts})
forecast  = model.predict(future_df)

predictions = np.maximum(forecast["yhat"].values, 0)
pred_lower  = np.maximum(forecast["yhat_lower"].values, 0)
pred_upper  = np.maximum(forecast["yhat_upper"].values, 0)

print(f"Hour-by-hour forecast for {future_ts[0]:%A, %Y-%m-%d}:")
for ts, p in zip(future_ts, predictions):
    tag = "  <- likely empty" if p < 5 else ""
    print(f"  {ts:%a %m-%d %H:%M} : {p:5.1f} people{tag}")


# %% [markdown]
# ## 6. Pitch chart — history + forecast + AC-shutdown trigger

# %%
SHUTDOWN_THRESHOLD = 5     # < 5 people = effectively empty
shutdown_ts = None
for i in range(len(predictions) - 1):
    if predictions[i] < SHUTDOWN_THRESHOLD and predictions[i + 1] < SHUTDOWN_THRESHOLD:
        shutdown_ts = future_ts[i] - timedelta(minutes=10)
        break

fig, ax = plt.subplots(figsize=(14, 5))

# Show last 3 days of history for visual context
recent = df.iloc[-72:]
ax.plot(recent["timestamp"], recent["occupancy"],
        color="#2c3e50", lw=1.4, label="History (last 3 days)")

# Forecast line
ax.plot(future_ts, predictions, color="#e74c3c", lw=2.4,
        label="Prophet forecast (next 24 h)")

# Confidence band (80th percentile)
ax.fill_between(future_ts, pred_lower, pred_upper,
                alpha=0.15, color="#e74c3c", label="80% confidence band")

# Shutdown trigger marker
if shutdown_ts is not None:
    ax.axvline(shutdown_ts, color="#27ae60", ls="--", lw=2.2,
               label=f"AC SHUTDOWN @ {shutdown_ts.strftime('%H:%M')}")
    ax.annotate(
        f"AC OFF\n{shutdown_ts.strftime('%a %H:%M')}",
        xy=(shutdown_ts, ax.get_ylim()[1] * 0.85),
        xytext=(10, 0), textcoords="offset points",
        color="#27ae60", weight="bold", fontsize=11,
    )

ax.set_title(
    f"Rewatt x Prophet — Occupancy Forecast for {future_ts[0]:%A %Y-%m-%d}, "
    f"Lecture Hall B",
    fontsize=13, weight="bold")
ax.set_ylabel("People present")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast.png")
plt.show()


# %% [markdown]
# ## 7. The line you read out during the pitch

# %%
if shutdown_ts is not None:
    next_busy_ts = None
    for i, p in enumerate(predictions):
        if future_ts[i] > shutdown_ts and p >= SHUTDOWN_THRESHOLD:
            next_busy_ts = future_ts[i]
            break

    msg = (
        f"\n>>> ACTION: Lecture Hall B will be empty from "
        f"{shutdown_ts.strftime('%H:%M on %A')}"
    )
    if next_busy_ts is not None:
        msg += f" until {next_busy_ts.strftime('%H:%M')}"
    msg += " - schedule AC shutdown.\n"
    print(msg)
else:
    print("\n>>> No multi-hour empty window detected in the next 24 h.")
