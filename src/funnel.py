import pandas as pd
import matplotlib.pyplot as plt
import os

IN = "data/events_clean.csv"
OUT = "reports"
os.makedirs(OUT, exist_ok=True)

events = pd.read_csv(IN, parse_dates=['event_time'])

funnel_events = events[events['event_type'].isin(['view', 'cart', 'purchase'])]

funnel = (
    funnel_events.groupby('event_type')['user_id']
    .nunique()
    .reindex(['view', 'cart', 'purchase'])
)

# Plot Funnel chart
plt.figure(figsize=(8, 5))
plt.bar(funnel.index, funnel.values)
plt.title('User Funnel: View → Add to Cart → Purchase')
plt.xlabel('Event Type')
plt.ylabel('Unique Users')
plt.tight_layout()
plt.savefig(f'{OUT}/funnel_chart.png', bbox_inches='tight')
plt.close()

print(funnel)
print("Funnel chart saved to reports/funnel_chart.png")
