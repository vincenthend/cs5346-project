import { useEffect, useRef } from 'react'
import 'ol/ol.css'
import Map from 'ol/Map'

interface Props {
  map: Map
}

function BackgroundMap(props: Props) {
  const { map } = props
  const mapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (mapRef.current) {
      map.setTarget(mapRef.current)
    }

    // Cleanup function
    return () => {
      map.setTarget()
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