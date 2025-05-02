import { Col, Layout, Row, Space, Typography } from 'antd'
import React, { useState } from 'react'
import { LayerToggle, LayerType } from '../../types'
import AreaCard from '../AreaCard'
import LayerCard from '../LayerCard'
import VisualizationMap from '../VisualizationMap'
import { Polygon } from 'ol/geom'
import { Collection, Feature } from 'ol'

const { Header, Content } = Layout

function AppContainer() {
  const [areas, setAreas] = useState<Collection<Feature<Polygon>>>(new Collection<Feature<Polygon>>())
  const [selectedArea, setSelectedArea] = useState()
  const [toggle, setToggle] = React.useState<LayerToggle>({ [LayerType.TAXI_LOCATION]: true })

  return (
    <>
      <Layout>
        <Header style={{ display: 'flex', alignItems: 'center', backgroundColor: '#000f1e', justifyContent: 'center' }}>
          <Typography.Text style={{ color: 'white', textAlign: 'center' }}>
            The data in this version of visualization is not using the LIVE version. The source code for this visualization is available <Typography.Link href={"https://github.com/vincenthend/cs5346-project"}>here</Typography.Link>
          </Typography.Text>
        </Header>
        <Header style={{ display: 'flex', alignItems: 'center' }}>
          <Typography.Text style={{ color: 'white', fontSize: '20px', fontWeight: 500 }}>
            Taxi Board
          </Typography.Text>
        </Header>
        <Content style={{ background: '#1a1a1a' }}>
          <Row wrap={false}>
            <Col flex={'auto'}>
              <VisualizationMap toggle={toggle} onSelect={setSelectedArea} onLoad={setAreas} />
            </Col>
            <Col span={4} style={{ padding: '8px' }}>
              <Space direction={'vertical'} style={{width: '100%'}}>
                <AreaCard feature={selectedArea} areas={areas} />
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
