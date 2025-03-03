import BackgroundMap from './components/BackgroundMap'
import LayerCard from './components/LayerCard'

function App() {
  return <>
    <BackgroundMap draggable={false} />
    <LayerCard />
  </>
}

export default App 