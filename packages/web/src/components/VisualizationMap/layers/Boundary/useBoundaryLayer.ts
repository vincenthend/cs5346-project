import { FeatureLike } from 'ol/Feature'
import { Vector } from 'ol/layer'
import BaseLayer from 'ol/layer/Base'
import Map from 'ol/Map'
import VectorSource from 'ol/source/Vector'
import { Fill, Stroke, Style } from 'ol/style'
import React, { useEffect, useRef } from 'react'
import useBoundaryLocationCollection from './useBoundaryLocationCollection.ts'

function useBoundaryLayer(map: Map, enabled?: boolean) {
  const mapLayer = useRef<BaseLayer>()
  const [boundaries] = useBoundaryLocationCollection(true)

  const enable = React.useCallback(() => {
    const geomStyle = (feature: FeatureLike) => {
      return [
        new Style({
          stroke: new Stroke({
            color: 'rgba(255,112,112, 0.5)',
            width: 2,
          }),
          fill: new Fill({
            color: 'rgba(255, 112, 112, 0.05)',
          }),
        }),
      ]
    }

    if (boundaries) {
      const vectorSource = new VectorSource({
        features: boundaries,
      })
      console.log(vectorSource)
      mapLayer.current = new Vector({
        source: vectorSource,
        style: geomStyle,
      })
      map.addLayer(mapLayer.current)
    }
  }, [boundaries])

  const disable = React.useCallback(() => {
    if (mapLayer.current) {
      map.removeLayer(mapLayer.current)
      mapLayer.current = undefined
    }
  }, [])

  useEffect(() => {
    if (enabled) {
      enable()
    } else {
      disable()
    }

    return () => {
      disable()
    }
  }, [enabled, enable, disable])
}

export default useBoundaryLayer
