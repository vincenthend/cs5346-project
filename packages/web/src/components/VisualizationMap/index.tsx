import { Heatmap } from 'ol/layer'
import TileLayer from 'ol/layer/Tile'
import { fromLonLat } from 'ol/proj'
import OSM from 'ol/source/OSM'
import VectorSource from 'ol/source/Vector'
import View from 'ol/View'
import { useEffect } from 'react'
import useMap from '../../hooks/useMap.ts'
import useTaxiLocationCollection from '../../hooks/useTaxiLocationCollection.ts'
import BackgroundMap from '../BackgroundMap'

const SINGAPORE_COORDS = fromLonLat([103.8198, 1.3521])

function VisualizationMap() {
  const [map] = useMap({
    layers: [
      new TileLayer({
        source: new OSM(),
      }),
    ],
    view: new View({
      center: SINGAPORE_COORDS,
      zoom: 12,
    }),
  })
  const [locations] = useTaxiLocationCollection()
  useEffect(() => {
    let locationLayer: Heatmap | undefined
    if (locations) {
      locationLayer = new Heatmap({ source: new VectorSource({ features: locations }) })
      map.addLayer(locationLayer)
    }

    return () => {
      if (locationLayer) {
        map.removeLayer(locationLayer)
      }
    }
  }, [locations])

  return <BackgroundMap map={map} />
}

export default VisualizationMap