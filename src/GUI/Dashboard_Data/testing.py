import pandas as pd

# Define the start and end date
start_date = "2016-01-01 00:00"
end_date = "2016-12-31 23:45"

# Create a date range with 15-minute intervals
date_range = pd.date_range(start=start_date, end=end_date, freq='15T')

# Create a DataFrame with the date range
date_time_2016 = pd.DataFrame({
    'Time Step': range(len(date_range)),
    'Date': date_range.date,
    'Time': date_range.time
})
print(date_time_2016)
# Display the first few rows of the DataFrame
target_date_time = "2016-01-01 00:00"
index = date_time_2016[
    (date_time_2016['Date'] == pd.to_datetime(target_date_time).date()) &
    (date_time_2016['Time'] == pd.to_datetime(target_date_time).time())
].index
# Print the index
print("index",index)
#print(all_date)

import plotly.graph_objects as go

# Sample data
x_data = ['2016-01-01 00:00:00', '2016-01-01 00:15:00', 3, 4, 5]
x_data = x_data_strings = [str(item) for item in x_data]
y_data = [10, 11, 12, 13, 14]

# Create a scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_data,  # x-axis data
    y=y_data,  # y-axis data
    mode='markers',  # 'lines', 'markers', or 'lines+markers'
    name='Sample Data',  # Name of the trace
    marker=dict(
        color='blue',  # Marker color
        size=10,  # Marker size
        symbol='circle'  # Marker symbol
    )
))

# Update layout
fig.update_layout(
    title='Scatter Plot Example',
    xaxis_title='X Axis',
    yaxis_title='Y Axis'
)

# Show the plot
fig.show()