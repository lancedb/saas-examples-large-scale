import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      include: ['@lancedb/lancedb'],
      globals: {
        Buffer: true,
        global: true,
        process: true,
      },
    })
  ],
  build: {
    rollupOptions: {
      external: ['@lancedb/lancedb']
    }
  },
  optimizeDeps: {
    exclude: ['@lancedb/lancedb']
  }
})