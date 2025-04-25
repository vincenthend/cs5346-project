import { Button, Card, Descriptions, Divider, Progress, Space, Typography } from 'antd'
import { Collection, Feature } from 'ol'
import { getCenter } from 'ol/extent'
import { MultiPolygon, Point, Polygon } from 'ol/geom'
import React from 'react'

function calculatePolygonDistance(feature1: Feature<Polygon| MultiPolygon>, feature2: Feature<Polygon | MultiPolygon>): number {
  const geom1 = feature1.getGeometry()!
  const geom2 = feature2.getGeometry()!
  let center1;
  if (geom1 instanceof Polygon) {
    center1 = getCenter(geom1.getExtent());
  } else if (geom1 instanceof MultiPolygon) {
    // Get the extent of the entire MultiPolygon
    center1 = getCenter(geom1.getExtent());
  } else {
    console.warn('Unsupported geometry type for geom1');
    return Infinity;
  }

  let center2;
  if (geom2 instanceof Polygon) {
    center2 = getCenter(geom2.getExtent());
  } else if (geom2 instanceof MultiPolygon) {
    center2 = getCenter(geom2.getExtent());
  } else {
    console.warn('Unsupported geometry type for geom2');
    return Infinity;
  }

  const coords1 = center1;
  const coords2 = center2;
  const dx = coords2[0] - coords1[0];
  const dy = coords2[1] - coords1[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function AreaCard({ feature, areas }: { feature: Feature<MultiPolygon>, areas: Collection<Feature<Polygon>> }) {


  const recommendations = React.useMemo(() => {
    if (!areas || !feature) return []
    // Get top 3 areas closest to feature
    // sort areas by distance of area to feature
    const sortedAreas = Array.from(areas).sort((a: Feature<Polygon>, b: Feature<Polygon>) => {
      const distanceA = calculatePolygonDistance(a, feature)
      const distanceB = calculatePolygonDistance(b, feature)
      return distanceA - distanceB // ascending order of di
    })
    const top3Areas = sortedAreas.map((x: Feature<Polygon>) => ({area: x.get('Description'), available: x.get('taxi_count'), demand_level: x.get('demand_count') / x.get('taxi_count') })).filter(x => x.demand_level < 1 && x.area !== feature.get('Description')).slice(0, 3)
    return top3Areas
  }, [areas, feature])

  
  if (!feature) return null
  return (
    <Space direction={'vertical'}>
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
        <Progress percent={100} success={{percent: (feature.get('taxi_count')) / feature.get('demand_count') * 100}} status={'exception'} showInfo={false} />
        
      </Space>
    </Card>
    {feature.get('taxi_count') < feature.get('demand_count') && 
    (<Card title={'Reroute Taxi'} size={'small'}>
        
        <div>
          <Typography.Text>Recommended Reroute</Typography.Text>
          <div style={{paddingLeft: 16}}>
            <ul>
              {recommendations.map((item, index) => (
                <li key={index}>{item.area} ({item.available} Available)</li>
              ))}
            </ul>
          </div>
          <Button type={'primary'} style={{ width: '100%' }}>
            Command Reroute Taxi
          </Button>
        </div>
    </Card>)}
    </Space>
  )
}

export default AreaCard
