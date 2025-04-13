import { LayerToggle, LayerType } from '../../types'
import BackgroundMap from '../BackgroundMap'
import useBoundaryLayer from './layers/Boundary/useBoundaryLayer.ts'
import useDemandLayer from './layers/Demand/useDemandLayer.ts'
import useSingaporeMap from './layers/SingaporeMap/useSingaporeMap.ts'
import useTaxiLocationLayer from './layers/TaxiLocation/useTaxiLocationLayer.ts'
import './styles.css'
import useWeatherLayer from './layers/Weather/useWeatherLayer.ts'

interface Props {
  toggle: LayerToggle
}

function VisualizationMap(props: Props) {
  const { toggle } = props
  const [map] = useSingaporeMap()
  useBoundaryLayer(map, true)
  //TODO: Merge these layers
  useTaxiLocationLayer(map, toggle[LayerType.TAXI_LOCATION])
  useDemandLayer(map, true)
  useWeatherLayer(map, toggle[LayerType.WEATHER])

  return <BackgroundMap map={map} />
}

export default VisualizationMap
