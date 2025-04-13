import { Collection, Feature } from 'ol'
import { GeoJSON } from 'ol/format'
import { Polygon } from 'ol/geom'
import { useEffect, useState } from 'react'
import useSWR from 'swr'
import Apis from '../../../../api'
import { ApiResponse } from '../../../../types'

function useBoundaryLocationCollection(enabled?: boolean): [Collection<Feature<Polygon>>] {
  const [locations, setLocations] = useState(new Collection<Feature<Polygon>>())
  const { data } = useSWR<ApiResponse<any>>(enabled && Apis.Boundary.getBoundary(), {
    revalidateOnFocus: false,
  })

  useEffect(() => {
    if (data) {
      setLocations(() => {
        const geojsonFormat = new GeoJSON()
        const _data = geojsonFormat.readFeatures(data.data, {
          featureProjection: 'EPSG:3857', // The projection of your map
          dataProjection: 'EPSG:4326', // The projection of your GeoJSON data (usually WGS 84)
        })
        return _data as Feature<Polygon>[]
      })
    }
  }, [data])

  return [locations]
}

export default useBoundaryLocationCollection
