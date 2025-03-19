import { Vector } from 'ol/layer'
import BaseLayer from 'ol/layer/Base'
import Map from 'ol/Map'
import VectorSource from 'ol/source/Vector'
import { Circle, Fill, Style } from 'ol/style'
import React, { useEffect, useRef } from 'react'
import useTaxiLocationCollection from '../../hooks/useTaxiLocationCollection.ts'

function useTaxiLocationLayer(map: Map, enabled?: boolean) {
  const [locations] = useTaxiLocationCollection()
  const locationLayer = useRef<BaseLayer>()

  const enable = React.useCallback(() => {
    const pointStyle = () => {
      const zoom = map.getView().getZoom() // Get the current zoom level

      return new Style({
        image: new Circle({
          radius: (zoom ?? 1) / 8,
          fill: new Fill({
            color: '#FFB121',
          }),
        }),
      })
    }

    if (locations) {
      locationLayer.current = new Vector({
        source: new VectorSource({ features: locations }),
        style: pointStyle,
      })
      map.addLayer(locationLayer.current)
    }
  }, [locations])

  const disable = React.useCallback(() => {
    if (locationLayer.current) {
      map.removeLayer(locationLayer.current)
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
