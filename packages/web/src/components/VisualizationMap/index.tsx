import { LayerToggle, LayerType } from '../../types'
import BackgroundMap from '../BackgroundMap'
import useSingaporeMap from './useSingaporeMap.ts'
import useTaxiLocationLayer from './useTaxiLocationLayer.ts'
import './styles.css'

interface Props {
  toggle: LayerToggle
}

function VisualizationMap(props: Props) {
  const { toggle } = props
  const [map] = useSingaporeMap()
  useTaxiLocationLayer(map, toggle[LayerType.TAXI_LOCATION])

  return <BackgroundMap map={map} />
}

export default VisualizationMap
