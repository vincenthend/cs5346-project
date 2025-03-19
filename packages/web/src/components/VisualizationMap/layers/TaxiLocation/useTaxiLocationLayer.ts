import { Vector } from 'ol/layer'
import BaseLayer from 'ol/layer/Base'
import Map from 'ol/Map'
import VectorSource from 'ol/source/Vector'
import { Circle, Fill, Style } from 'ol/style'
import React, { useEffect, useRef } from 'react'
import useTaxiLocationCollection from './useTaxiLocationCollection.ts'

function useTaxiLocationLayer(map: Map, enabled?: boolean) {
  const [locations] = useTaxiLocationCollection(enabled)
  const mapLayer = useRef<BaseLayer>()

  const enable = React.useCallback(() => {
    const pointStyle = (_, resolution) => {
      return new Style({
        image: new Circle({
          radius: Math.min(20 / resolution, 3),
          fill: new Fill({
            color: '#FFB121',
          }),
        }),
      })
    }

    if (locations) {
      mapLayer.current = new Vector({
        source: new VectorSource({ features: locations }),
        style: pointStyle,
      })
      map.addLayer(mapLayer.current)
    }
  }, [locations])

  const disable = React.useCallback(() => {
    if (mapLayer.current) {
      map.removeLayer(mapLayer.current)
      mapLayer.current = undefined
    }
  }, [])

  useEffect(() => {
    if(enabled) {
      enable()
    } else {
      disable()
    }

    return () => {
      disable()
    }
  }, [enabled, enable, disable])
}

export default useTaxiLocationLayer
