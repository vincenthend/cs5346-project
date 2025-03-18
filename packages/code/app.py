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
    print(data)
    return {
        "success": True,
        "data": {
            "count": data['features'][0]['properties']['taxi_count'],
            "locations": data['features'][0]['geometry']['coordinates'],
            "timestamp": data['features'][0]['properties']['timestamp']
        }
    }

if __name__ == '__main__':
    app.run(port=8000)
