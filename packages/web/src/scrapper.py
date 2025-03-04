import os
import requests
import csv
import time
import json
from datetime import datetime, timedelta
import pandas as pd

# API endpoints and headers
taxi_url = "https://datamall2.mytransport.sg/ltaodataservice/Taxi-Availability"
rain_url = "https://api.data.gov.sg/v1/environment/rainfall"
traffic_incidents_url = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
headers = {
    "AccountKey": "89nzBN7BT5uL/b9GeWJJuw==",
    "accept": "application/json"
}

# Function to format taxi ID with leading zeros
def format_taxi_id(index):
    return f"{index + 1:05d}"

# Function to fetch taxi availability data
def get_taxi_availability():
    response = requests.get(taxi_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve taxi data. Status code: {response.status_code}")
        return None

# Function to fetch rain data
def get_rain_data():
    response = requests.get(rain_url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(f"Failed to retrieve rain data. Status code: {response.status_code}")
        return None

# Function to fetch traffic incidents data
def get_traffic_incidents():
    response = requests.get(traffic_incidents_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve traffic incidents data. Status code: {response.status_code}")
        return None

# Function to process rain data
def process_rain_data(rain):
    lst = []
    stations = rain['metadata']['stations']
    readings = rain['items'][0]['readings']
    timestamp = rain['items'][0]['timestamp']

    readings_dict = {reading['station_id']: reading['value'] for reading in readings}

    for station in stations:
        station_id = station['id']
        station_name = station['name']
        lat = station['location']['latitude']
        long = station['location']['longitude']
        rain_value = readings_dict.get(station_id, 0)  # Default to 0 if no reading

        lst.append([station_id, station_name, rain_value, lat, long])

    rain_df = pd.DataFrame(lst, columns=['id', 'name', 'rain_value', 'latitude', 'longitude'])
    return rain_df, timestamp

# Function to create CSV files for each dataset
def create_csv_file(file_prefix):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = "./" + file_prefix
    os.makedirs(directory, exist_ok=True)
    file_name = f"{directory}/{file_prefix}_{current_time}.csv"
    return file_name

# Function to save taxi data to CSV
def save_taxi_data_to_csv(file_name, taxi_data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Taxi ID", "Longitude", "Latitude"])

        for index, taxi in enumerate(taxi_data['value']):
            taxi_id = format_taxi_id(index)
            taxi_long = taxi["Longitude"]
            taxi_lat = taxi["Latitude"]
            writer.writerow([timestamp, taxi_id, taxi_long, taxi_lat])

    print(f"Taxi data saved at {timestamp} in {file_name}")


# Function to save rain data to CSV
def save_rain_data_to_csv(file_name, rain_df, rain_timestamp):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Station Name", "Rain Value", "Station Longitude", "Station Latitude"])

        for _, rain_station in rain_df.iterrows():
            writer.writerow(
                [rain_timestamp, rain_station['name'], rain_station['rain_value'], rain_station['longitude'],
                 rain_station['latitude']])

    print(f"Rain data saved at {rain_timestamp} in {file_name}")


def save_traffic_incidents_data_to_csv(file_name, traffic_incidents_data):
    retrieval_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writerow([
                "Timestamp", "Latitude", "Longitude", "Message", "Type", "Incident Timestamp"
            ])

        for incident in traffic_incidents_data['value']:
            incident_lat = incident.get("Latitude", "")
            incident_long = incident.get("Longitude", "")
            incident_message = incident.get("Message", "")
            incident_type = incident.get("Type", "")
            incident_timestamp = ""

            # Parse the incident timestamp from the message
            start_idx = incident_message.find('(')
            end_idx = incident_message.find(')', start_idx)
            if start_idx != -1 and end_idx != -1:
                date_part = incident_message[start_idx + 1:end_idx]  # Extract '9/11'
                
                # Extract time_part: characters after ')' up to the next space
                end_of_time_idx = incident_message.find(' ', end_idx + 1)
                if end_of_time_idx != -1:
                    time_part = incident_message[end_idx + 1:end_of_time_idx]
                    # Extract the rest of the message after time_part
                    cleaned_message = incident_message[end_of_time_idx + 1:].strip()
                else:
                    # No space found; time_part goes to the end of the string
                    time_part = incident_message[end_idx + 1:]
                    cleaned_message = ""  # No message content after time

                # Parse date and time
                day_str, month_str = date_part.split('/')
                # Swap day and month to get correct date
                day = day_str.zfill(2)
                month = month_str.zfill(2)
                incident_date = f"2024-{month}-{day}"
                incident_time = time_part.strip()
                incident_timestamp = f"{incident_date} {incident_time}"
            else:
                cleaned_message = incident_message  # If no date/time, keep the original message

            # Write all data to CSV
            writer.writerow([
                retrieval_timestamp, incident_lat, incident_long, cleaned_message, incident_type,
                incident_timestamp
            ])

    print(f"Traffic incidents data saved at {retrieval_timestamp} in {file_name}")





# Main function to run the script for 24 hours
if __name__ == "__main__":
    taxi_file_name = create_csv_file("taxi_availability")
    rain_file_name = create_csv_file("rain_data")
    traffic_file_name = create_csv_file("traffic_incidents")

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=24)
    # end_time = start_time + timedelta(minutes=5)  # Run for 5 minutes for testing

    while datetime.now() < end_time:
        taxi_data = get_taxi_availability()
        rain_data = get_rain_data()
        traffic_incidents_data = get_traffic_incidents()

        if taxi_data:
            save_taxi_data_to_csv(taxi_file_name, taxi_data)

        if rain_data:
            rain_df, rain_timestamp = process_rain_data(rain_data)
            save_rain_data_to_csv(rain_file_name, rain_df, rain_timestamp)

        if traffic_incidents_data:
            save_traffic_incidents_data_to_csv(traffic_file_name, traffic_incidents_data)

        if datetime.now() >= start_time + timedelta(minutes=10):
        # if datetime.now() >= start_time + timedelta(seconds=30):  # Reduce time for testing
            taxi_file_name = create_csv_file("taxi_availability")
            rain_file_name = create_csv_file("rain_data")
            traffic_file_name = create_csv_file("traffic_incidents")
            start_time = datetime.now()

        time.sleep(10)

    print("Data collection completed for 24 hours.")
