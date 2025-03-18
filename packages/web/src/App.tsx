import { SWRConfig } from 'swr'
import LayerCard from './components/LayerCard'
import VisualizationMap from './components/VisualizationMap'
import { appFetch } from './utils/fetch.ts'

function App() {
  return (
    <>
      <SWRConfig value={{ fetcher: appFetch }}>
        <VisualizationMap draggable={false} />
        <LayerCard />
      </SWRConfig>
    </>
  )
}

export default App