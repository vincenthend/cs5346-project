import { useEffect, useRef } from 'react'
import 'ol/ol.css'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import { fromLonLat } from 'ol/proj'

function BackgroundMap() {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<Map | null>(null)

  useEffect(() => {
    if (mapRef.current && !mapInstance.current) {
      // Singapore coordinates
      const singaporeCoords = fromLonLat([103.8198, 1.3521])
      
      // Create the map
      mapInstance.current = new Map({
        target: mapRef.current,
        layers: [
          new TileLayer({
            source: new OSM()
          })
        ],
        view: new View({
          center: singaporeCoords,
          zoom: 12
        })
      })
    }

    // Cleanup function
    return () => {
      if (mapInstance.current) {
        mapInstance.current.setTarget(null)
        mapInstance.current = null
      }
    }
  }, [])

  return (
    <div 
      ref={mapRef} 
      style={{ 
        width: '100vw', 
        height: '100vh', 
        margin: 0, 
        padding: 0 
      }} 
    />
  )
}

export default BackgroundMap