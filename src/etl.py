import pandas as pd

events = pd.read_csv("data/events.csv", parse_dates=['event_time'])

events.columns = events.columns.str.lower()
events['event_type'] = events['event_type'].str.lower().fillna('')
events['event_date'] = events['event_time'].dt.date

funnel_events = events[events['event_type'].isin(['view','cart','purchase'])]
funnel_events.drop_duplicates(subset=['user_id','event_type','event_date'], inplace=True)
print(funnel_events.isna().sum())

funnel_events.to_csv("data/events_clean.csv", index=False)

