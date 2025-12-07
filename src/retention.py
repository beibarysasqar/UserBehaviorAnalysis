import pandas as pd
import matplotlib.pyplot as plt
import os

IN = "data/events_clean.csv"
OUT = "reports"
os.makedirs(OUT, exist_ok=True)

events = pd.read_csv(IN, parse_dates=['event_time'])
events['event_date'] = events['event_time'].dt.date

# First event per user
first_event = events.groupby('user_id')['event_time'].min().reset_index()
first_event.rename(columns={'event_time': 'first_event'}, inplace=True)

events = events.merge(first_event, on='user_id')
events['days_since_first'] = (events['event_time'] - events['first_event']).dt.days

# Cohort retention
df_ret = events.groupby(['user_id', 'days_since_first']).size().reset_index(name='events')

# Pivot for retention matrix
retention = df_ret.pivot_table(index='days_since_first', columns='user_id', values='events', fill_value=0)

# Sum users per day
cohort = (retention > 0).sum(axis=1)

# Plot retention curve
plt.figure(figsize=(10,5))
plt.plot(cohort.index, cohort.values)
plt.xlabel('Days since first event')
plt.ylabel('Active users')
plt.title('Retention Curve')
plt.tight_layout()
plt.savefig(f'{OUT}/retention_heatmap.png', bbox_inches='tight')
plt.close()
print("Retention Heatmap saved to reports/retention_heatmap.png")
