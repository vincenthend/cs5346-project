import { Button, Card, Descriptions, Space } from 'antd'
import { Feature } from 'ol'
import { Polygon } from 'ol/geom'
import React from 'react'

function AreaCard({ feature }: { feature: Feature<Polygon> }) {
  if (!feature) return null
  return (
    <Card size="small" title={'Operations'}>
      {/*{feature && <pre>{JSON.stringify(feature)}</pre>}*/}
      <Space direction={'vertical'}>
        <Descriptions
          column={2}
          items={[
            {
              label: 'Selected Area',
              children: feature.get('Description'),
              span: 2,
            },
            {
              label: 'Demand',
              children: feature.get('demand_count'),
            },
            {
              label: 'Available',
              children: feature.get('taxi_count'),
            },
          ]}
        />
        <Button type={'primary'} style={{ width: '100%' }}>
          Command Reroute Taxi
        </Button>
      </Space>
    </Card>
  )
}

export default AreaCard
