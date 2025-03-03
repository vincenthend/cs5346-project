import { Button, Card, Typography } from "antd";

function LayerDisplay({onSelect, name, icon}: {onSelect?: (layer: string) => void, name: string, icon: string }) {
    return <div>
        <Button onClick={() => onSelect?.(name)} type="text" style={{ display: 'flex', alignItems: 'center', gap: 8, flexDirection: 'column', height: 'fit-content', width: 64, padding: 8 }}>
            <div style={{ width: 96, height: 96, backgroundColor: icon, borderRadius: 4 }}></div>
            <Typography.Text>{name}</Typography.Text>
        </Button>
    </div>
}

function LayerCard({ onSelect }: { onSelect?: (layer: string) => void }) {
  return <Card style={{ position: 'absolute', bottom: 24, right: 16, zIndex: 1000 }} size="small">
    <div style={{ display: 'flex', flexDirection: 'row', gap: 8 }}>
        <LayerDisplay onSelect={onSelect} name="Layer 1" icon="#000" />
        <LayerDisplay onSelect={onSelect} name="Layer 2" icon="#000" />
        <LayerDisplay onSelect={onSelect} name="Layer 3" icon="#000" />
    </div>
  </Card>
}

export default LayerCard