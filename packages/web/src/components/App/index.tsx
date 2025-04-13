import { Col, Layout, Row, Space, Typography } from 'antd'
import React, { useState } from 'react'
import { LayerToggle, LayerType } from '../../types'
import AreaCard from '../AreaCard'
import LayerCard from '../LayerCard'
import VisualizationMap from '../VisualizationMap'

const { Header, Content } = Layout

function AppContainer() {
  const [selectedArea, setSelectedArea] = useState()
  const [toggle, setToggle] = React.useState<LayerToggle>({ [LayerType.TAXI_LOCATION]: true })

  return (
    <>
      <Layout>
        <Header style={{ display: 'flex', alignItems: 'center' }}>
          <Typography.Text style={{ color: 'white', fontSize: '20px', fontWeight: 500 }}>
            Taxi Board
          </Typography.Text>
        </Header>
        <Content style={{ background: '#1a1a1a' }}>
          <Row wrap={false}>
            <Col flex={'auto'}>
              <VisualizationMap toggle={toggle} onSelect={setSelectedArea} />
            </Col>
            <Col span={4} style={{ padding: '8px' }}>
              <Space direction={'vertical'} style={{width: '100%'}}>
                <AreaCard feature={selectedArea} />
                <LayerCard
                  value={toggle}
                  onChange={(layer, checked) => {
                    setToggle((val) => {
                      return { ...val, [layer]: checked }
                    })
                  }}
                />
              </Space>
            </Col>
          </Row>
        </Content>
      </Layout>
    </>
  )
}

export default AppContainer
