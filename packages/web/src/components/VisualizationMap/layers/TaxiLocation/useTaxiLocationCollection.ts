import { Collection, Feature } from 'ol'
import { Geometry, Point } from 'ol/geom'
import { fromLonLat } from 'ol/proj'
import { useEffect, useState } from 'react'
import useSWR from 'swr'
import Apis from '../../../../api'
import { ApiResponse } from '../../../../types'
import { TaxiData } from '../../../../types/taxi.ts'

function useTaxiLocationCollection(enabled?: boolean): [Collection<Feature<Point>>] {
  const [locations, setLocations] = useState(new Collection<Feature<Point>>())
  const { data } = useSWR<ApiResponse<TaxiData>>(enabled && Apis.Taxi.getLocations(), {
    refreshInterval: 30 * 1000,
    fetcher: () => import('../../../../data/get_taxi_locations.json').then((res) => res),
  })

  useEffect(() => {
    if (data) {
      setLocations(() => {
        return new Collection(
          data.data.locations.map((p) => {
            return new Feature({ geometry: new Point(fromLonLat(p)) })
          })
        )
      })
    }
  }, [data])

  return [locations]
}

export default useTaxiLocationCollection
