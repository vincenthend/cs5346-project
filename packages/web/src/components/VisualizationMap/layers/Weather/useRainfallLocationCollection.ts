import { Collection, Feature } from 'ol'
import { Geometry, Point } from 'ol/geom'
import { fromLonLat } from 'ol/proj'
import { useEffect, useState } from 'react'
import useSWR from 'swr'
import Apis from '../../../../api'
import { ApiResponse } from '../../../../types'
import { WeatherData } from '../../../../types/weather.ts'

function useRainfallLocationCollection(enabled?: boolean): [Collection<Feature<Geometry>>] {
  const [locations, setLocations] = useState(new Collection<Feature<Geometry>>())
  const { data } = useSWR<ApiResponse<WeatherData[]>>(enabled && Apis.Weather.getRainfallLocations(), {
    refreshInterval: 5 * 60 * 1000,
  })

  useEffect(() => {
    if (data) {
      setLocations(() => {
        return new Collection(
          data.data.map((p) => {
            const feature = new Feature({ geometry: new Point(fromLonLat(p.location)) })
            feature.set('data', p)
            return feature
          })
        )
      })
    }
  }, [data])

  return [locations]
}

export default useRainfallLocationCollection