import requests

def getTaxiLocations():
    url = "https://api.data.gov.sg/v1/transport/taxi-availability"
    response = requests.get(url)
    data = response.json()
    return {
        "count": data['features'][0]['properties']['taxi_count'],
        "locations": data['features'][0]['geometry']['coordinates'],
        "timestamp": data['features'][0]['properties']['timestamp']
    }