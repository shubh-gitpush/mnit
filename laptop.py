import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('synthetic_wtp_laptop_data.csv')

# Use the correct column names
features = ['Memory', 'Storage', 'CPU_class', 'Screen_size', 'year']
target = 'price'

# Prepare the data
X = df[features]
y = df[target]

# Train a regression model
model = LinearRegression()
model.fit(X, y)

# Base model specifications
base_specs = {
    'Memory': 16,
    'Storage': 512,
    'CPU_class': 1,
    'Screen_size': 14.0,
    'year': 2025
}
base_price = 111000

# Upgrade options
upgrades = {
    'add_16gb_memory':      ({'Memory': base_specs['Memory'] + 16}, 7000),
    'add_512gb_storage':    ({'Storage': base_specs['Storage'] + 512}, 5000),
    'upgrade_cpu':          ({'CPU_class': base_specs['CPU_class'] + 1}, 15000),
    'increase_screen_size': ({'Screen_size': 16.0}, 3000),
}

results = []

# Estimate profit for each upgrade
for name, (change, cost) in upgrades.items():
    new_specs = base_specs.copy()
    new_specs.update(change)
    specs_df = pd.DataFrame([new_specs])
    est_price = model.predict(specs_df)[0]
    gross_profit = est_price - base_price - cost
    results.append((name, gross_profit))

# Show top 2 upgrades
top_two = sorted(results, key=lambda x: x[1], reverse=True)[:2]
for name, profit in top_two:
    print(name, "total profit",round(profit))
