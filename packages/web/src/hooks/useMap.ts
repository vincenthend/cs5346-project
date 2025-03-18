import React from 'react'
import Map, { MapOptions } from 'ol/Map'

function useMap(init?: MapOptions) {
  const [mapInstance] = React.useState<Map>(new Map(init))
  return [mapInstance]
}

export default useMap