import { Collection, Feature } from 'ol'
import { Point } from 'ol/geom'
import { fromLonLat } from 'ol/proj'
import { useEffect, useState } from 'react'
import useSWR from 'swr'
import Apis from '../../../../api'
import { ApiResponse } from '../../../../types'

function useDemandCollection(enabled?: boolean): [Collection<Feature<Point>>] {
  const [locations, setLocations] = useState(new Collection<Feature<Point>>())
  const { data } = useSWR<
    ApiResponse<
      {
        grid_x: number
        grid_y: number
        taxi_count: number
      }[]
    >
  >(enabled && Apis.Demand.getDemand(), {
    revalidateOnFocus: false,
    fetcher: () => import('../../../../data/get_demand.json').then((res) => res),
  })

  useEffect(() => {
    if (data) {
      setLocations(() => {
        const _data = new Collection<Feature<Point>>()
        data.data.forEach((loc) => {
          _data.push(new Feature(new Point(fromLonLat([loc.grid_x / 100, loc.grid_y / 100]))))
        })
        return _data
      })
    }
  }, [data])

  return [locations]
}

export default useDemandCollection
