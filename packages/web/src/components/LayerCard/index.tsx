import { CarOutlined, CloudOutlined, LineChartOutlined } from '@ant-design/icons'
import { Card, Col, Row, Switch, Typography } from 'antd'
import React from 'react'
import { LayerToggle, LayerType } from '../../types'

function LayerDisplay({
  onChange,
  label,
  icon,
  value,
}: {
  onChange: (checked: boolean) => void
  label?: React.ReactNode
  icon: React.ReactNode
  value?: boolean
}) {
  return (
    <Row gutter={8} wrap={false}>
      <Col>{icon}</Col>
      <Col flex={'auto'}>
        <Typography.Text>{label}</Typography.Text>
      </Col>
      <Col>
        <Switch value={value} onChange={(checked) => onChange(checked)} />
      </Col>
    </Row>
  )
}

function LayerCard({
  onChange,
  value,
}: {
  onChange: (layer: string, checked: boolean) => void
  value: LayerToggle
}) {
  const layersData = [
    { name: LayerType.TAXI_LOCATION, label: 'Taxi Location', icon: <CarOutlined /> },
    { name: LayerType.WEATHER, label: 'Weather', icon: <CloudOutlined /> },
  ]

  return (
    <Card
      size="small"
      title={'Layers'}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {layersData.map((layer) => (
          <LayerDisplay
            key={layer.name}
            onChange={(val) => onChange(layer.name, val)}
            label={layer.label}
            icon={layer.icon}
            value={value[layer.name]}
          />
        ))}
      </div>
    </Card>
  )
}

export default LayerCard
