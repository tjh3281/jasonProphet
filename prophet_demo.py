"""
Rewatt x Prophet Demo
=====================
Uses four weeks of fake hourly occupancy data for "Lecture Hall B" to:
  - Forecast the next 7 days  -> forecast_week.png
  - Forecast a single day     -> forecast_day.png

Prophet understands day-of-week and Malaysia public holidays natively.

HOW TO RUN (VS Code)
--------------------
1. Open a terminal in VS Code (Ctrl+`) and activate the virtual environment:
     .venv\Scripts\activate          # Windows
     source .venv/bin/activate       # macOS / Linux
2. Install dependencies:
     pip install -r requirements.txt
3. Run: python prophet_demo.py
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
# ## 2. Generate four weeks of fake occupancy data
#
# No timetable — just random headcounts that follow a realistic pattern:
#   Weekday daytime (8am-6pm): 30-120 people
#   Weekday nighttime        : 0-15  people
#   Weekend (all hours)      : 0-20  people

# %%
START    = datetime(2026, 4, 22, 0, 0)
DAYS     = 28
HOURS    = DAYS * 24
CAPACITY = 120

records = []
for h in range(HOURS):
    ts = START + timedelta(hours=h)
    wd, hh = ts.weekday(), ts.hour

    if wd >= 5:                          # weekend
        occ = np.random.randint(0, 20)
    elif 8 <= hh < 18:                   # weekday daytime
        occ = np.random.randint(30, CAPACITY)
    else:                                # weekday nighttime
        occ = np.random.randint(0, 15)

    records.append({"timestamp": ts, "occupancy": float(occ)})

df = pd.DataFrame(records)
print(df.head(12).to_string(index=False))
print(f"\nTotal hours of history: {len(df)}")


# %% [markdown]
# ## 3. Visualise the history (sanity check)

# %%
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.plot(df["timestamp"], df["occupancy"], lw=0.8, color="#2c3e50")
ax.set_title("Lecture Hall B — four weeks of fake hourly occupancy",
             weight="bold")
ax.set_ylabel("People present")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("history.png")
plt.show()


# %% [markdown]
# ## 4. Fit Prophet model

# %%
prophet_df = df.rename(columns={"timestamp": "ds", "occupancy": "y"})

model = Prophet(
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode="additive",
    interval_width=0.80,
)
model.add_country_holidays(country_name="MY")  # Malaysia public holidays
model.fit(prophet_df)
print("Prophet model fitted.")


# %% [markdown]
# ## 5. Forecast — single day (next 24 hours)

# %%
last_ts    = df["timestamp"].iloc[-1]
future_day = [last_ts + timedelta(hours=i + 1) for i in range(24)]

forecast_day = model.predict(pd.DataFrame({"ds": future_day}))
pred_day     = np.maximum(forecast_day["yhat"].values, 0)
lower_day    = np.maximum(forecast_day["yhat_lower"].values, 0)
upper_day    = np.maximum(forecast_day["yhat_upper"].values, 0)

print(f"\nHour-by-hour forecast for {future_day[0]:%A, %Y-%m-%d}:")
for ts, p in zip(future_day, pred_day):
    tag = "  <- likely empty" if p < 5 else ""
    print(f"  {ts:%H:%M} : {p:5.1f} people{tag}")

hour_labels = [ts.strftime("%H:%M") for ts in future_day]
x_pos = range(24)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x_pos, pred_day, color="#e74c3c", alpha=0.7, label="Forecast")
ax.fill_between(x_pos, lower_day, upper_day,
                alpha=0.2, color="#e74c3c", label="80% confidence band")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(hour_labels, rotation=45, ha="right", fontsize=8)
ax.set_title(f"Lecture Hall B — Single Day Forecast: {future_day[0]:%A, %d %B %Y}",
             fontsize=13, weight="bold")
ax.set_ylabel("People present")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("forecast_day.png")
plt.show()
print("Saved: forecast_day.png")


# %% [markdown]
# ## 6. Forecast — whole week (next 7 days)

# %%
future_week = [last_ts + timedelta(hours=i + 1) for i in range(7 * 24)]

forecast_week = model.predict(pd.DataFrame({"ds": future_week}))
pred_week     = np.maximum(forecast_week["yhat"].values, 0)
lower_week    = np.maximum(forecast_week["yhat_lower"].values, 0)
upper_week    = np.maximum(forecast_week["yhat_upper"].values, 0)

print("\nHour-by-hour forecast for the next 7 days:")
for day in range(7):
    day_ts    = future_week[day * 24]
    day_preds = pred_week[day * 24:(day + 1) * 24]
    day_times = future_week[day * 24:(day + 1) * 24]
    print(f"\n  --- {day_ts:%A, %Y-%m-%d} ---")
    for ts, p in zip(day_times, day_preds):
        tag = "  <- likely empty" if p < 5 else ""
        print(f"    {ts:%H:%M} : {p:5.1f} people{tag}")

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(future_week, pred_week, color="#2980b9", lw=2,
        label="Prophet forecast (7 days)")
ax.fill_between(future_week, lower_week, upper_week,
                alpha=0.15, color="#2980b9", label="80% confidence band")

# Day boundary lines
for day in range(1, 7):
    boundary = future_week[day * 24]
    ax.axvline(boundary, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(boundary, ax.get_ylim()[1] * 0.95,
            boundary.strftime("%a"), ha="center", fontsize=8, color="gray")

ax.set_title("Lecture Hall B — 7-Day Occupancy Forecast",
             fontsize=13, weight="bold")
ax.set_ylabel("People present")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_week.png")
plt.show()
print("Saved: forecast_week.png")
