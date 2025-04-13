import { Feature } from 'ol'
import { Polygon } from 'ol/geom'
import { LayerToggle, LayerType } from '../../types'
import BackgroundMap from '../BackgroundMap'
import useSingaporeMap from './layers/SingaporeMap/useSingaporeMap.ts'
import useTaxiLocationLayer from './layers/TaxiLocation/useTaxiLocationLayer.tsx'
import useWeatherLayer from './layers/Weather/useWeatherLayer.ts'
import './styles.css'

interface Props {
  toggle: LayerToggle
  onSelect: (feature: Feature<Polygon>) => void
}

function VisualizationMap(props: Props) {
  const { toggle } = props
  const [map] = useSingaporeMap()
  useTaxiLocationLayer(map, toggle[LayerType.TAXI_LOCATION], props.onSelect)
  useWeatherLayer(map, toggle[LayerType.WEATHER])

  return (
    <>
      <BackgroundMap map={map} />
    </>
  )
}

export default VisualizationMap
