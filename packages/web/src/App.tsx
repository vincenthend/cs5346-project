import { useState } from 'react'
import { Button, Layout, Typography } from 'antd'

const { Header, Content } = Layout
const { Title } = Typography

function App() {
  const [count, setCount] = useState(0)

  return (
    <Layout className="min-h-screen">
      <Header className="flex items-center">
        <Title level={3} style={{ color: 'white', margin: 0 }}>
          Dataviz App
        </Title>
      </Header>
      <Content className="p-8">
        <div className="text-center">
          <Button type="primary" onClick={() => setCount(count + 1)}>
            Count is: {count}
          </Button>
        </div>
      </Content>
    </Layout>
  )
}

export default App 