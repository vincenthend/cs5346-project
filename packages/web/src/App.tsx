import React from 'react'
import { SWRConfig } from 'swr'
import LayerCard from './components/LayerCard'
import VisualizationMap from './components/VisualizationMap'
import { LayerToggle, LayerType } from './types'
import { appFetch } from './utils/fetch.ts'

function App() {
  const [toggle, setToggle] = React.useState<LayerToggle>({ [LayerType.TAXI_LOCATION]: true })

  return (
    <>
      <SWRConfig value={{ fetcher: appFetch }}>
        <VisualizationMap toggle={toggle} />
        <LayerCard
          value={toggle}
          onChange={(layer, checked) => {
            setToggle((val) => {
              return { ...val, [layer]: checked }
            })
          }}
        />
      </SWRConfig>
    </>
  )
}

export default App
