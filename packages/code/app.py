from flask import Flask, send_from_directory
import requests

app = Flask(__name__, static_folder='.')

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/')
def index():
    return {"success": True}

@app.route('/api/get_taxi_locations')
def getTaxiLocations():
    url = "https://api.data.gov.sg/v1/transport/taxi-availability"
    response = requests.get(url)
    data = response.json()
    return {
        "success": True,
        "data": {
            "count": data['features'][0]['properties']['taxi_count'],
            "locations": data['features'][0]['geometry']['coordinates'],
            "timestamp": data['features'][0]['properties']['timestamp']
        }
    }

#####

def merge_stations_and_items(data):
    # Create a dictionary to map station_id to station details for quick lookup
    stations_dict = {station['id']: station for station in data['metadata']['stations']}

    # Initialize an empty list to store merged items
    merged_items = []

    # Iterate through each item in the 'items' array
    for item in data['items']:
        timestamp = item['timestamp']
        readings = []

        # Process each reading within the current item
        for reading in item['readings']:
            station_id = reading['station_id']

            # Merge the reading with its corresponding station details
            if station_id in stations_dict:
                merged_reading = {
                    **stations_dict[station_id],
                    'location': [stations_dict[station_id]['location']['longitude'], stations_dict[station_id]['location']['latitude']],
                    'value': reading['value'],
                    'timestamp': timestamp
                }
                readings.append(merged_reading)

        # Append the merged readings to the list of items
        merged_items.append({
            'timestamp': timestamp,
            'readings': readings
        })

    # Return the updated data with merged stations and items
    return merged_items[0]['readings']

@app.route('/api/get_rainfall_locations')
def getRainfallLocations():
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    response = requests.get(url)
    data = response.json()
    return {
        "success": True,
        "data": merge_stations_and_items(data)
    }

if __name__ == '__main__':
    app.run(port=8000)
