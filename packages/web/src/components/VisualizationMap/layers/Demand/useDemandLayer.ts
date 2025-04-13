import { Collection, Feature } from 'ol'
import { getCenter } from 'ol/extent'
import { FeatureLike } from 'ol/Feature'
import { Point } from 'ol/geom'
import { Vector } from 'ol/layer'
import BaseLayer from 'ol/layer/Base'
import Map from 'ol/Map'
import VectorSource from 'ol/source/Vector'
import { Circle, Fill, Style, Text } from 'ol/style'
import React, { useEffect, useRef } from 'react'
import useBoundaryLocationCollection from '../Boundary/useBoundaryLocationCollection.ts'
import useDemandCollection from './useDemandCollection.ts'

function useDemandLayer(map: Map, enabled?: boolean) {
  const [locations] = useDemandCollection(enabled)
  const [boundaries] = useBoundaryLocationCollection(enabled)
  const mapLayer = useRef<BaseLayer>()

  const enable = React.useCallback(() => {
    const pointStyle = (feature: FeatureLike, resolution: number) => {
      const size = feature.get('count')
      return new Style({
        image: new Circle({
          radius: 12,
          fill: new Fill({
            color: '#FF0000',
          }),
        }),
        text: new Text({
          text: size.toString(),
          font: 'bold 12px Arial',
          fill: new Fill({
            color: '#000',
          }),
        }),
      })
    }

    if (locations && boundaries) {
      const clusterSource = new Collection<Feature<Point>>()
      boundaries.forEach((b) => {
        const polygonGeometry = b.getGeometry()!
        let count = 0
        locations.forEach((l) => {
          const point = l.getGeometry()!
          if (polygonGeometry!.intersectsCoordinate(point.getCoordinates())) {
            count++
          }
        })

        const feature = new Feature<Point>(new Point(getCenter(polygonGeometry.getExtent())))
        feature.setProperties({ count })
        clusterSource.push(feature)
      })

      mapLayer.current = new Vector({
        source: new VectorSource({ features: clusterSource }),
        style: pointStyle,
      })
      map.addLayer(mapLayer.current)
    }
  }, [locations, boundaries])

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

export default useDemandLayer
