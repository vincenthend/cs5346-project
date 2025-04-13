import { Heatmap } from 'ol/layer'
import Map from 'ol/Map'
import VectorSource from 'ol/source/Vector'
import React, { useEffect, useRef } from 'react'
import { mapValue } from '../../../../utils'
import useRainfallLocationCollection from './useRainfallLocationCollection.ts'

function createBlurredCirclePattern(radius, color, blurRadius) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const size = radius * 2 + blurRadius * 2
  canvas.width = size
  canvas.height = size

  ctx.beginPath()
  ctx.arc(radius + blurRadius, radius + blurRadius, radius, 0, 2 * Math.PI)
  ctx.fillStyle = color
  ctx.shadowColor = color
  ctx.shadowBlur = blurRadius
  ctx.fill()

  return ctx.createPattern(canvas, 'repeat')
}

function getRadius(map: Map) {
  const resolution = map.getView().getResolution()
  return 1500 / resolution
}

function useWeatherLayer(map: Map, enabled?: boolean) {
  const [locations] = useRainfallLocationCollection(enabled)
  const mapLayer = useRef<Heatmap>()

  useEffect(() => {
    const updateRadius = () => {
      if (mapLayer.current) {
        mapLayer.current.setRadius(getRadius(map))
        mapLayer.current.setBlur(getRadius(map))
      }
    }
    map.getView().on('change:resolution', updateRadius)
    return () => {
      map.getView().un('change:resolution', updateRadius)
    }
  }, [enabled])

  const enable = React.useCallback(() => {
    if (locations) {
      mapLayer.current = new Heatmap({
        source: new VectorSource({ features: locations }),
        radius: getRadius(map),
        blur: getRadius(map),
        gradient: [`rgba(0,217,255,0.1)`, `rgba(0, 217, 255, 0.15)`],
        weight: (feature) => {
          return mapValue(feature.get('data').value, 0, 1, 0, 2)
        },
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

export default useWeatherLayer
