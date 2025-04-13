import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from geopy.distance import geodesic
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import time
import pandas as pd
import math

from weather import getRainfallLocations
from taxi import getTaxiLocations

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, kernel_size=3):
        super(STGCN, self).__init__()

        # Temporal convolution layer
        self.temporal_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=hidden_channels,
                                       kernel_size=(kernel_size, 1),
                                       padding=(kernel_size // 2, 0))

        # Graph convolution layers
        self.spatial_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.spatial_conv2 = GCNConv(hidden_channels, hidden_channels)

        # Output layer
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weights):
        # Input x is (batch_size, num_timesteps, num_nodes, in_channels)

        # Reshape input for temporal convolution
        batch_size, num_timesteps, num_nodes, in_channels = x.shape
        x = x.permute(0, 3, 2, 1)  # Change to (batch_size, in_channels, num_nodes, num_timesteps)

        # Apply temporal convolution
        x = self.temporal_conv(x)  # Output shape: (batch_size, hidden_channels, num_nodes, num_timesteps)
        x = F.relu(x)

        # Reshape back after temporal conv for spatial graph conv
        x = x.permute(0, 3, 2, 1)  # Change to (batch_size, num_timesteps, num_nodes, hidden_channels)

        # Flatten timesteps and batch for spatial convolution
        x = x.reshape(-1, num_nodes, x.shape[-1])  # Shape: (batch_size * num_timesteps, num_nodes, hidden_channels)

        # Apply spatial convolutions using the graph structure
        x = x.view(-1, x.shape[-1])  # Flatten for GCNConv: (batch_size * num_timesteps * num_nodes, hidden_channels)
        x = self.spatial_conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = self.spatial_conv2(x, edge_index, edge_weights)

        # Reshape and aggregate across timesteps
        x = x.view(batch_size, num_timesteps, num_nodes,
                   -1)  # Reshape back to (batch_size, num_timesteps, num_nodes, hidden_channels)

        # Output prediction layer
        x = self.fc(x)  # Shape: (batch_size, num_timesteps, num_nodes, out_channels)
        return x[:, -1]  # Return the prediction for the last timestep


def read_taxi_data():
    locations = getTaxiLocations()

    # Iterate through all files in the taxi folder
    data = []
    id = 1
    for location in locations["locations"]:
        data.append({
            "Timestamp": pd.to_datetime(time.time(), unit='s'),
            "Taxi ID": f"{id:05d}",
            "Longitude": location[0],
            "Latitude": location[1]
        })
        id += 1

    # Concatenate all the taxi data
    return pd.DataFrame.from_dict(data)


# Function to read and process weather data files, adding a timestamp from the filename
def read_weather_data():
    weatherdata = getRainfallLocations()

    data = []
    for w in weatherdata:
        data.append({
            "Timestamp": pd.to_datetime(time.time(), unit='s'),
            "Station Name": w['name'],
            "Rain Value": w['value'],
            "Station Longitude": w['location'][0],
            "Station Latitude": w['location'][1]
        })

    return pd.DataFrame.from_dict(data)


# Function to assign a region based on latitude and longitude
def assign_region(df, lat_col, lon_col, grid_size):
    # Create grid cell columns by dividing lat/lon by grid size
    df['grid_x'] = (df[lon_col] // grid_size).astype(int)
    df['grid_y'] = (df[lat_col] // grid_size).astype(int)
    return df


# Function to check if a neighbor grid has weather data
def has_weather_data(weather_data, grid_x, grid_y):
    return not weather_data[(weather_data['grid_x'] == grid_x) & (weather_data['grid_y'] == grid_y)].empty


# Function to get the min and max lat/lon from taxi data to define the region
def get_region_bounds(taxi_data, weather_data):
    min_lat = taxi_data['Latitude'].min()
    max_lat = taxi_data['Latitude'].max()
    min_lon = taxi_data['Longitude'].min()
    max_lon = taxi_data['Longitude'].max()
    min_lat_w = weather_data['Station Latitude'].min()
    max_lat_w = weather_data['Station Latitude'].max()
    min_lon_w = weather_data['Station Longitude'].min()
    max_lon_w = weather_data['Station Longitude'].max()

    min_lat = min(min_lat, min_lat_w)
    max_lat = max(max_lat, max_lat_w)
    min_lon = min(min_lon, min_lon_w)
    max_lon = max(max_lon, max_lon_w)

    return min_lat, max_lat, min_lon, max_lon


# Function to check if a grid is within the region bounds
def is_within_bounds(grid_x, grid_y, min_lat, max_lat, min_lon, max_lon, grid_size):
    grid_lat = grid_y * grid_size
    grid_lon = grid_x * grid_size
    return (min_lat - grid_size <= grid_lat <= max_lat + grid_size) and (
                min_lon - grid_size <= grid_lon <= max_lon + grid_size)


# Function to fill neighbors' grids with weather data, but only within the region bounds
def propagate_weather_data(weather_data, grid_size, min_lat, max_lat, min_lon, max_lon):
    # Iterate over unique timestamps
    for timestamp in weather_data['Timestamp'].unique():
        updated = True
        processed_grids = set()  # Set to track which grids have already been updated for this timestamp

        while updated:  # Keep looping until no changes are made
            updated = False

            # Extract weather data for the current timestamp
            weather_data_at_time = weather_data[weather_data['Timestamp'] == timestamp]

            # Iterate through grids that have weather data for this timestamp
            for index, row in tqdm(weather_data_at_time.iterrows(), total=weather_data_at_time.shape[0],
                                   desc=f"Propagating weather data for {timestamp}"):
                grid_x, grid_y = row['grid_x'], row['grid_y']
                neighbors = [(grid_x - 1, grid_y), (grid_x + 1, grid_y), (grid_x, grid_y - 1), (grid_x, grid_y + 1)]

                # Propagate to neighboring grids, but only if within bounds and not processed before
                for neighbor_x, neighbor_y in neighbors:
                    if is_within_bounds(neighbor_x, neighbor_y, min_lat, max_lat, min_lon, max_lon, grid_size):
                        if not has_weather_data(weather_data_at_time, neighbor_x, neighbor_y) and (
                        neighbor_x, neighbor_y) not in processed_grids:
                            # Copy weather data to the neighboring grid if it doesn't have any
                            new_row = row.copy()
                            new_row['grid_x'] = neighbor_x
                            new_row['grid_y'] = neighbor_y
                            processed_grids.add((neighbor_x, neighbor_y))  # Mark this grid as processed

                            # Add the new row to both the temporary and global weather dataframes
                            weather_data_at_time = pd.concat([weather_data_at_time, pd.DataFrame([new_row])],
                                                             ignore_index=True)
                            weather_data = pd.concat([weather_data, pd.DataFrame([new_row])], ignore_index=True)

                            updated = True  # If any grid was updated, set to True to continue the loop

    return weather_data


def align_taxi_weather_data(taxi_data, weather_data, time_diff_minutes=11, grid_size=0.01):
    # Assign taxis and weather stations to regions based on latitude and longitude
    taxi_data = assign_region(taxi_data, 'Latitude', 'Longitude', grid_size)
    weather_data = assign_region(weather_data, 'Station Latitude', 'Station Longitude', grid_size)

    # Get the bounds of the region from the taxi data
    min_lat, max_lat, min_lon, max_lon = get_region_bounds(taxi_data, weather_data)

    # Propagate weather data to neighboring grids without weather information (but only within the region bounds)
    weather_data = propagate_weather_data(weather_data, grid_size, min_lat, max_lat, min_lon, max_lon)

    # Create columns for storing weather info in taxi data
    taxi_data['Rain Value'] = None
    taxi_data['weather_time'] = pd.NaT

    # Iterate through unique regions and align data within each region
    for (grid_x, grid_y) in tqdm(list(taxi_data[['grid_x', 'grid_y']].drop_duplicates().itertuples(index=False)),
                                 desc="Aligning regions"):
        taxis_in_region = taxi_data[(taxi_data['grid_x'] == grid_x) & (taxi_data['grid_y'] == grid_y)]
        weather_in_region = weather_data[(weather_data['grid_x'] == grid_x) & (weather_data['grid_y'] == grid_y)]

        for index, taxi_row in taxis_in_region.iterrows():
            nearest_weather = weather_in_region.iloc[
                (weather_in_region['Timestamp'] - taxi_row['Timestamp']).abs().argsort()[:1]]

            if not nearest_weather.empty:
                time_diff = abs((nearest_weather['Timestamp'].values[0] - taxi_row[
                    'Timestamp']).total_seconds()) / 60  # Convert to minutes
                if time_diff <= time_diff_minutes:
                    taxi_data.at[index, 'Rain Value'] = nearest_weather['Rain Value'].values[0]
                    taxi_data.at[index, 'weather_time'] = nearest_weather['Timestamp'].values[0]

    return taxi_data


def read_merge_taxi_data(file_path):
    all_taxi_data = []

    # Iterate through all files in the taxi folder

    taxi_data = pd.read_csv(file_path)

    all_taxi_data.append(taxi_data)

    # Concatenate all the taxi data
    return pd.concat(all_taxi_data, ignore_index=True)


def aggregate_data(merged_data, grid_size, time_interval):
    aggregated_data = merged_data.groupby(['grid_x', 'grid_y', 'Timestamp']).agg({
        'Taxi ID': 'count',  # Count the number of taxis
        'Rain Value': 'mean'  # Average rain intensity
    }).reset_index()

    # Rename columns for clarity
    aggregated_data.rename(columns={'Taxi ID': 'available_taxi_count'}, inplace=True)

    return aggregated_data


def preprocess_timestamp(time_series):
    # Ensure the input is in datetime format
    time_series = pd.to_datetime(time_series, format='%Y-%m-%d %H:%M:%S')

    # Extract the time of day and convert it to seconds since midnight
    time_of_day = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second

    # Add offset for 2024-10-21
    date_offset = (time_series.dt.date == pd.to_datetime('2024-10-21').date()) * 86400

    # Return the total timestamp, considering the date offset
    total_timestamp = time_of_day + date_offset

    return total_timestamp


# Create sequences of timesteps for each grid cell
def create_time_windows(data, window_size):
    """
    Create time windows of size `window_size` for each grid, including a placeholder for
    the target `available_taxi_count` in the last window frame.

    Args:
        data: DataFrame with time-ordered taxi data (must include 'Timestamp' and 'available_taxi_count' columns).
        window_size: Number of timesteps to include in each sequence.

    Returns:
        A tensor of shape [num_sequences, window_size, num_features] and a target tensor.
    """
    # Get the complete set of unique grids (grid_x, grid_y)
    complete_grids = data[['grid_x', 'grid_y']].drop_duplicates().reset_index(drop=True)
    num_nodes = complete_grids.shape[0]  # Number of unique grid cells

    # Features: include 'grid_x', 'grid_y', 'Rain Value', and 'available_taxi_count' for all timestamps
    feature_columns = [col for col in data.columns if
                       col not in ['Timestamp', 'available_taxi_count', 'grid_x', 'grid_y']]
    num_features = len(feature_columns)

    sequences = []
    targets = []

    unique_timestamps = sorted(data['Timestamp'].unique())

    for idx in tqdm(range(window_size, len(unique_timestamps))):
        window_data = []

        # Prepare the data for the past `window_size` timesteps
        for i in range(window_size):
            timestamp = unique_timestamps[idx + i - window_size]
            window_frame = data[data['Timestamp'] == timestamp]

            # Merge the complete grids to ensure missing grids are added
            merged_frame = complete_grids.merge(window_frame, on=['grid_x', 'grid_y'], how='left')

            # For the last timestep in the window, set 'available_taxi_count' to 0 (placeholder)
            # if i == window_size - 1:
            #     merged_frame['available_taxi_count'] = 0  # Placeholder for the target value

            merged_frame = merged_frame[feature_columns + ['available_taxi_count']].fillna(
                0)  # Fill missing values with 0

            window_data.append(merged_frame.values)

        # Convert window_data to a numpy array
        window_data = np.array(window_data)

        # Target value for the next timestep (predict the taxi count for the next timestep)
        target_timestamp = unique_timestamps[idx]
        target_frame = data[data['Timestamp'] == target_timestamp][['grid_x', 'grid_y', 'available_taxi_count']]

        # Merge target_frame with complete grids to align target with features
        merged_target_frame = complete_grids.merge(target_frame, on=['grid_x', 'grid_y'], how='left')
        merged_target_frame['available_taxi_count'] = merged_target_frame['available_taxi_count'].fillna(
            0)  # Fill missing target values with 0

        target_values = merged_target_frame['available_taxi_count'].values  # Extract target values

        # Append the window and target to their respective lists
        sequences.append(window_data)
        targets.append(target_values)

    # Convert the lists to tensors
    return torch.tensor(np.array(sequences), dtype=torch.float), torch.tensor(np.array(targets), dtype=torch.float)


# Now, use the function to create the sequences and target data


# Example edge index preparation (assume you have a function to create spatial/temporal edges)

def create_temporal_edges(num_nodes, window_size):
    """
    Create temporal edges for each node across consecutive timesteps.

    Args:
        num_nodes: Number of nodes (grids) at each timestep.
        window_size: Number of timesteps in each sequence.

    Returns:
        edge_index: Tensor of shape [2, num_temporal_edges].
    """
    temporal_edges = []

    for t in range(window_size - 1):
        for node in range(num_nodes):
            # Connect node at time t with the same node at time t+1
            temporal_edges.append([node + t * num_nodes, node + (t + 1) * num_nodes])

    edge_index_temporal = torch.tensor(temporal_edges, dtype=torch.long).t().contiguous()
    return edge_index_temporal


def create_flow_based_edges(timestamp_index, movements, num_nodes, window_size):
    """
    Create flow-based edges from inferred taxi movements between grid cells over time.

    Args:
        movements: List of inferred movements (start_grid, end_grid, change_in_taxi_count, Timestamp).
        num_nodes: Number of nodes (grids) per timestep.
        window_size: Number of timesteps in the sequence.

    Returns:
        flow_edges: Tensor of shape [2, num_flow_edges] representing flow-based edges.
    """
    flow_edges = []
    edge_weights = []

    for movement in movements:
        start_grid = movement['start_grid']
        end_grid = movement['end_grid']
        t = timestamp_index[movement['Timestamp']]  # Map timestamp to index

        # Check if t + 1 exists in aggregated_data's timestamps
        if t + 1 >= window_size:
            continue

        # Calculate the node index for the start and end grid at time t and t+1
        start_idx = start_grid[0] + start_grid[1] * num_nodes + t * num_nodes
        end_idx = end_grid[0] + end_grid[1] * num_nodes + (t + 1) * num_nodes

        # Create an edge from the start grid (at time t) to the end grid (at time t+1)
        flow_edges.append([start_idx, end_idx])
        edge_weights.append(movement['change_in_taxi_count'])

    # Convert to tensors
    flow_edges = torch.tensor(flow_edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    return flow_edges, edge_weights


# Modify the graph creation function to support batches of temporal data
def create_grid_graph_temporal(timestamp_index, data, window_size, edge_index_spatial, movements):
    """
    Create combined spatial, temporal, and flow-based edges for ST-GCN.

    Args:
        data: DataFrame with taxi data.
        window_size: Number of timesteps in each sequence.
        edge_index_spatial: Spatial edges.
        movements: List of inferred taxi movements between grids.

    Returns:
        edge_index_combined: Tensor of shape [2, num_edges] representing combined spatial, temporal, and flow-based edges.
    """
    num_nodes = data[['grid_x', 'grid_y']].drop_duplicates().shape[0]  # Number of nodes at each timestep

    # Create temporal edges
    edge_index_temporal = create_temporal_edges(num_nodes, window_size)

    # Create flow-based edges from inferred movements
    edge_index_flow, edge_weights_flow = create_flow_based_edges(timestamp_index, movements, num_nodes, window_size)

    spatial_weights = torch.ones(edge_index_spatial.size(1), dtype=torch.float)
    temporal_weights = torch.ones(edge_index_temporal.size(1), dtype=torch.float)

    #######################################
    ### Adjust the graph struction below ##
    #######################################

    # spatial-temporal-flow
    edge_index_combined = torch.cat([edge_index_spatial, edge_index_temporal, edge_index_flow], dim=1)
    edge_weights_combined = torch.cat([spatial_weights, temporal_weights, edge_weights_flow], dim=0)

    # spatial-temporal
    # edge_index_combined = torch.cat([edge_index_spatial, edge_index_temporal], dim=1)
    # edge_weights_combined = torch.cat([spatial_weights, temporal_weights], dim=0)

    # temporal
    # edge_index_combined = torch.cat([edge_index_temporal], dim=1)
    # edge_weights_combined = torch.cat([temporal_weights], dim=0)

    return edge_index_combined, edge_weights_combined


def create_spatial_edges(data):
    """
    Create spatial edges between neighboring grids based on grid_x and grid_y.

    Args:
        data: DataFrame with columns ['grid_x', 'grid_y'].

    Returns:
        edge_index: Tensor of shape [2, num_edges] representing the spatial edges.
    """
    edge_index = []
    unique_grids = data[['grid_x', 'grid_y']].drop_duplicates()

    for _, row in unique_grids.iterrows():
        x, y = row['grid_x'], row['grid_y']

        # Define neighbors (4-connected grid structure)
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for nx, ny in neighbors:
            # Check if the neighbor exists in the data
            if ((unique_grids['grid_x'] == nx) & (unique_grids['grid_y'] == ny)).any():
                source_idx = unique_grids[(unique_grids['grid_x'] == x) & (unique_grids['grid_y'] == y)].index[0]
                target_idx = unique_grids[(unique_grids['grid_x'] == nx) & (unique_grids['grid_y'] == ny)].index[0]

                edge_index.append([source_idx, target_idx])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def infer_taxi_movements(taxi_counts, grid_size):
    """
    Infer taxi movements between grids by comparing taxi counts across grids and consecutive timesteps.

    Args:
        taxi_counts: DataFrame with columns ['grid_x', 'grid_y', 'Timestamp', 'taxi_count'].
        grid_size: The size of the grid cells in degrees.

    Returns:
        movements: List of inferred movements (start_grid, end_grid, change_in_taxi_count).
    """
    movements = []

    # Sort by timestamp to compare consecutive timesteps
    taxi_counts = taxi_counts.sort_values(by=['Timestamp', 'grid_x', 'grid_y'])

    # Iterate over unique timestamps
    timestamps = sorted(taxi_counts['Timestamp'].unique())
    print(timestamps)

    # Define neighbors relative to a grid cell
    neighbors_offset = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Top, bottom, left, right neighbors

    for t in tqdm(range(len(timestamps) - 1)):
        curr_timestep = taxi_counts[taxi_counts['Timestamp'] == timestamps[t]]
        next_timestep = taxi_counts[taxi_counts['Timestamp'] == timestamps[t + 1]]

        # Iterate over each grid in the current timestep
        for index, row in curr_timestep.iterrows():
            start_grid = (row['grid_x'], row['grid_y'])
            start_taxi_count = row['available_taxi_count']

            # Find the taxi count in the next timestep for the same grid (start grid)
            next_taxi_count = next_timestep[(next_timestep['grid_x'] == start_grid[0]) &
                                            (next_timestep['grid_y'] == start_grid[1])]

            # Skip if there's no corresponding data in the next timestep
            if next_taxi_count.empty:
                continue

            next_taxi_count = next_taxi_count['available_taxi_count'].values[0]

            # If the taxi count has decreased in this grid, we assume taxis have moved out
            if start_taxi_count > next_taxi_count:
                # Check for potential destinations (neighboring grids)
                for neighbor_offset in neighbors_offset:
                    neighbor_x = start_grid[0] + neighbor_offset[0]
                    neighbor_y = start_grid[1] + neighbor_offset[1]

                    # Find the neighbor's taxi count in the current and next timesteps
                    neighbor_curr = curr_timestep[(curr_timestep['grid_x'] == neighbor_x) &
                                                  (curr_timestep['grid_y'] == neighbor_y)]
                    neighbor_next = next_timestep[(next_timestep['grid_x'] == neighbor_x) &
                                                  (next_timestep['grid_y'] == neighbor_y)]

                    # If the neighbor exists in both timesteps, compare their taxi counts
                    if not neighbor_curr.empty and not neighbor_next.empty:
                        neighbor_taxi_count_curr = neighbor_curr['available_taxi_count'].values[0]
                        neighbor_taxi_count_next = neighbor_next['available_taxi_count'].values[0]

                        # If the neighbor's taxi count increased, infer movement
                        if neighbor_taxi_count_next > neighbor_taxi_count_curr:
                            movements.append({
                                'start_grid': start_grid,
                                'end_grid': (neighbor_x, neighbor_y),
                                'change_in_taxi_count': neighbor_taxi_count_next - neighbor_taxi_count_curr,
                                'Timestamp': row['Timestamp']  # Include the timestamp for rain access
                            })

    return movements

model = STGCN(num_nodes=473, in_channels=2, hidden_channels=64, out_channels=1)
checkpoint = torch.load('checkpoint.pth')  # Replace with the actual checkpoint path
model.load_state_dict(checkpoint['model_state_dict'])

def predict_demand():
    taxi_data = read_taxi_data()
    weather_data = read_weather_data()

    merged_data = align_taxi_weather_data(taxi_data, weather_data)
    test_data = merged_data
    grid_size = 0.01
    time_interval = 15  # Time interval in minutes
    aggregated_test_data = aggregate_data(test_data, grid_size, time_interval)
    aggregated_test_data = aggregated_test_data.sort_values(by=['grid_x', 'grid_y', 'Timestamp'])
    aggregated_test_data['Timestamp'] = pd.to_datetime(aggregated_test_data['Timestamp'])
    unique_timestamps = sorted(aggregated_test_data['Timestamp'].unique())
    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(unique_timestamps)}
    movements_test = infer_taxi_movements(aggregated_test_data, grid_size)

    aggregated_test_data = aggregated_test_data.sort_values(by='Timestamp')

    # Create spatial edges
    edge_index_spatial = create_spatial_edges(aggregated_test_data)

    # Combine spatial, temporal, and flow-based edges for ST-GCN
    window_size = 5  # Example window size
    edge_index_combined_test, edge_weights_combined_test = create_grid_graph_temporal(timestamp_index, aggregated_test_data, window_size,
                                                                                      edge_index_spatial, movements_test)

    # Prepare the Data object for ST-GCN
    x_sequences, y_sequences = create_time_windows(aggregated_test_data, window_size)
    print(f"x_sequences shape: {x_sequences.shape}")  # Should be [num_sequences, window_size, num_nodes, num_features]
    print(f"y_sequences shape: {y_sequences.shape}")  # Should be [num_sequences, num_nodes]
    scaler = StandardScaler()
    x_sequences = scaler.fit_transform(x_sequences.reshape(-1, x_sequences.shape[-1])).reshape(x_sequences.shape)

    test_data = Data(x=torch.tensor(x_sequences, dtype=torch.float),
                     edge_index=edge_index_combined_test,
                     y=torch.tensor(y_sequences, dtype=torch.float))

    model.eval()
    predictions = []
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index, test_data.edge_attr)
        predictions.append(out.cpu().numpy())

    # Convert predictions to a NumPy array
    predictions = np.array(predictions).squeeze()
    print("Predictions shape:", predictions.shape)  # Expected shape: [num_samples, num_nodes]

    # 10. Convert predictions to JSON format
    latest_prediction = predictions[-1]  # Take the last predicted frame
    grid_info = aggregated_test_data[['grid_x', 'grid_y']].drop_duplicates().reset_index(drop=True)

    return [
        {"grid_x": int(grid_info.iloc[i]['grid_x']), "grid_y": int(grid_info.iloc[i]['grid_y']),
         "taxi_count":  math.ceil(latest_prediction[i])}
        for i in range(len(grid_info))
    ]
    # return True
