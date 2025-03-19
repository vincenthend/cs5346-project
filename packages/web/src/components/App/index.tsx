import { Layout, Typography } from 'antd'
import React from 'react'
import LayerCard from '../LayerCard'
import VisualizationMap from '../VisualizationMap'
import { LayerToggle, LayerType } from '../../types'

const { Header, Content } = Layout

function AppContainer() {
  const [toggle, setToggle] = React.useState<LayerToggle>({ [LayerType.TAXI_LOCATION]: true })

  return (
    <>
      <Layout>
        <Header style={{ display: 'flex', alignItems: 'center' }}>
          <Typography.Text style={{ color: 'white', fontSize: '20px', fontWeight: 500 }}>Taxi Board</Typography.Text>
        </Header>
        <Content>
          <VisualizationMap toggle={toggle} />
          <LayerCard
            value={toggle}
            onChange={(layer, checked) => {
              setToggle((val) => {
                return { ...val, [layer]: checked }
              })
            }}
          />
        </Content>
      </Layout>
    </>
  )
}

export default AppContainer
