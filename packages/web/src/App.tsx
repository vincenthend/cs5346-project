import { ConfigProvider } from 'antd'
import { SWRConfig } from 'swr'
import AppContainer from './components/App'
import RerouteModal from './components/RerouteModal'
import { appFetch } from './utils/fetch.ts'

const THEME = { components: { Layout: { headerHeight: 48, headerPadding: 24 } } }

function App() {
  return (
    <>
      <SWRConfig value={{ fetcher: appFetch }}>
        <ConfigProvider theme={THEME}>
          <AppContainer />
          <RerouteModal />
        </ConfigProvider>
      </SWRConfig>
    </>
  )
}

export default App
