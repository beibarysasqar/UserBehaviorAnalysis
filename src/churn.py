import pandas as pd
import os

IN = "data/events_clean.csv"
OUT = "reports"
os.makedirs(OUT, exist_ok=True)

events = pd.read_csv(IN, parse_dates=['event_time'])
events['event_time'] = events['event_time'].dt.tz_localize(None)
events['event_date'] = events['event_time'].dt.date

# Snapshot date (last day in dataset)
snapshot_date = pd.to_datetime(events['event_date'].max())

# Last Activity + Churn Label
last_activity = (
    events.groupby('user_id')['event_time']
    .max()
    .reset_index()
    .rename(columns={'event_time': 'last_time'})
)

last_activity['days_since_last'] = (
    snapshot_date - pd.to_datetime(last_activity['last_time'])
).dt.days

CHURN_DAYS = 14
last_activity['churned'] = (last_activity['days_since_last'] >= CHURN_DAYS).astype(int)

# First Activity
first_event = (
    events.groupby('user_id')['event_time']
    .min()
    .reset_index()
    .rename(columns={'event_time': 'first_time'})
)

events = events.merge(first_event, on='user_id')
events['days_from_first'] = (events['event_time'] - events['first_time']).dt.days


# FEATURES
# Number of events in the first 7 days
feat_7d = (
    events[events['days_from_first'] <= 7]
    .groupby('user_id')['event_type']
    .count()
    .reset_index()
    .rename(columns={'event_type': 'cnt_7d'})
)

# Total events
total_events = (
    events.groupby('user_id')['event_type']
    .count()
    .reset_index()
    .rename(columns={'event_type': 'total_events'})
)

# Purchase flag (1 if user has at least one purchase)
has_purchase = (
    (events[events['event_type'] == 'purchase']
    .groupby('user_id')['event_type']
    .count() > 0)
    .astype(int)
    .reset_index()
    .rename(columns={'event_type': 'has_purchase'})
)


# Merge User Features 
users = (
    last_activity
        .merge(feat_7d, on='user_id', how='left')
         .merge(total_events, on='user_id', how='left')
         .merge(has_purchase, on='user_id', how='left')
)

users[['cnt_7d', 'total_events', 'has_purchase']] = (
    users[['cnt_7d', 'total_events', 'has_purchase']].fillna(0)
)


# Reports
print("\n=== Churn Rate ===")
print("Overall churn rate:", round(users["churned"].mean(), 3))

print("\n=== Feature Averages by Churn ===")

print("\nEvents in first 7 days:")
print(users.groupby("churned")["cnt_7d"].mean())

print("\nTotal events:")
print(users.groupby("churned")["total_events"].mean())

print("\nShare of users with purchases:")
print(users.groupby("churned")["has_purchase"].mean())


# Save final dataset
users.to_csv(f"{OUT}/churn_dataset.csv", index=False)

print("\nFile saved:", f"{OUT}/churn_dataset.csv")