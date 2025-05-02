import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    __MAPBOX_TOKEN: JSON.stringify(process.env.MAPBOX_TOKEN)
  },
  server: {
    port: 3000
  }
}) 