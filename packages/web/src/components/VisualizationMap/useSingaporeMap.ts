import { Zoom } from 'ol/control'
import TileLayer from 'ol/layer/Tile'
import { fromLonLat } from 'ol/proj'
import { XYZ } from 'ol/source'
import View from 'ol/View'
import env from '../../constants/env.ts'
import useMap from '../../hooks/useMap.ts'

const MAPBOX_TOKEN = env.MAPBOX_TOKEN
const MAPBOX_URL = `https://api.mapbox.com/styles/v1/vincenthendrha/cm8g5my47007101sded43brxa/tiles/512/{z}/{x}/{y}?access_token=${MAPBOX_TOKEN}`
const SINGAPORE_COORDS = fromLonLat([103.8198, 1.3521])
const MIN_SINGAPORE = fromLonLat([103.6, 1.2])
const MAX_SINGAPORE = fromLonLat([104.05, 1.55])
const SINGAPORE_EXTENT = [MIN_SINGAPORE[0], MIN_SINGAPORE[1], MAX_SINGAPORE[0], MAX_SINGAPORE[1]]

function useSingaporeMap() {
  const [map] = useMap({
    layers: [
      new TileLayer({
        // source: new OSM(),

        source: new XYZ({
          url: MAPBOX_URL,
          tileSize: 512, // Default tile size
        }),
      }),
    ],
    view: new View({
      center: SINGAPORE_COORDS,
      zoom: 10,
      extent: SINGAPORE_EXTENT,
    }),
    controls: [
      new Zoom({
        className: 'ol-zoom-bottom-right', // Custom CSS class
      }),
    ],
  })
  return [map]
}

export default useSingaporeMap
