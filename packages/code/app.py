from flask import Flask, send_from_directory
import requests
import json
from taxi import getTaxiLocations
from weather import getRainfallLocations
from model import predict_demand

app = Flask(__name__, static_folder='.')

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/')
def index():
    return {"success": True}


@app.route('/api/get_taxi_locations')
def getTaxiLocationsApi():
    url = "https://api.data.gov.sg/v1/transport/taxi-availability"
    response = requests.get(url)
    data = response.json()
    return {
        "success": True,
        "data": getTaxiLocations()
    }

@app.route('/api/get_rainfall_locations')
def getRainfallLocationsApi():
    return {
        "success": True,
        "data": getRainfallLocations()
    }

@app.route('/api/get_boundary_area')
def getAreaBoundary():
    boundary_data = json.load(open('./data/boundary.geojson'))
    return {
        "success": True,
        "data": boundary_data
    }

@app.route('/api/get_demand')
def getDemand():
    # demand = predict_demand()
    demand = json.load(open('./data/predicted_taxi_availability.json'))
    return {
        "success": True,
        "data": demand,
    }

if __name__ == '__main__':
    app.run(port=5000)
