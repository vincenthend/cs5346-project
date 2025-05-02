import chroma from 'chroma-js'
import { Collection, Feature } from 'ol'
import { FeatureLike } from 'ol/Feature'
import { Polygon } from 'ol/geom'
import { Vector } from 'ol/layer'
import BaseLayer from 'ol/layer/Base'
import Map from 'ol/Map'
import VectorSource from 'ol/source/Vector'
import { Fill, Stroke, Style, Text } from 'ol/style'
import React, { useEffect, useRef } from 'react'
import useBoundaryLocationCollection from '../Boundary/useBoundaryLocationCollection.ts'
import useDemandCollection from './useDemandCollection.ts'
import useTaxiLocationCollection from './useTaxiLocationCollection.ts'

const scaleNeg = chroma.scale(['#FFFFFF05', '#FF000077']).domain([1, 1.5])
const scalePos = chroma.scale(['#00FF0077', '#FFFFFF05']).domain([0, 1])

function useTaxiLocationLayer(
  map: Map,
  enabled?: boolean,
  onSelect?: (feature: Feature<Polygon>) => void,
  onLoad?: (areas: Collection<Feature<Polygon>>) => void
) {
  const [demands] = useDemandCollection(enabled)
  const [locations] = useTaxiLocationCollection(enabled)
  const [boundaries] = useBoundaryLocationCollection(enabled)

  const mapLayer = useRef<BaseLayer>()

  const enable = React.useCallback(() => {
    // setup actions
    map.on('singleclick', function (e) {
      const features = map.getFeaturesAtPixel(e.pixel)
      if (features.length === 0) {
        onSelect(null)
      } else {
        const feature: Feature<Polygon> = features.find((x) => x.get('demand_count'))
        onSelect(feature)
      }
    })

    // setup layers
    const pointStyle = (feature: FeatureLike, resolution: number) => {
      const demandCount = feature.get('demand_count')
      const taxiCount = feature.get('taxi_count')
      const name = feature.get('Description')

      const demandLevel = demandCount / taxiCount

      return new Style({
        stroke: new Stroke({
          color: '#ffffff55',
          lineDash: [10, 10],
          width: 2,
        }),
        fill: new Fill({
          color: `${isNaN(demandLevel) ? 'transparent' : demandLevel > 1 ? scaleNeg(demandLevel).hex('rgba') : scalePos(demandLevel).hex('rgba')}`,
        }),
        text: new Text({
          text: `${name}`,
          font: '12px Arial',
          fill: new Fill({color: '#FFF'})
        }),
      })
    }

    const dataJson = []
    if (locations && boundaries) {
      boundaries.forEach((b) => {
        const polygonGeometry = b.getGeometry()!
        let taxi_count = 0
        let demand_count = 0
        demands.forEach((d) => {
          const point = d.getGeometry()!
          if (polygonGeometry!.intersectsCoordinate(point.getCoordinates())) {
            demand_count++
          }
        })
        locations.forEach((l) => {
          const point = l.getGeometry()!
          if (polygonGeometry!.intersectsCoordinate(point.getCoordinates())) {
            taxi_count++
          }
        })

        b.setProperties({ demand_count, taxi_count })
          dataJson.push([b.get('Description'), demand_count, taxi_count])
      })
      console.log(JSON.stringify(dataJson))

      onLoad(boundaries)

      mapLayer.current = new Vector({
        source: new VectorSource({ features: boundaries }),
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

export default useTaxiLocationLayer
