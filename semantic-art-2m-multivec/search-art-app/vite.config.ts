import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      globals: {
        Buffer: true,
        global: true,
        process: true,
      },
      protocolImports: true
    })
  ],

  optimizeDeps: {
    exclude: [
      '@lancedb/lancedb' // Add this line
      // You might technically exclude '@lancedb/lancedb-darwin-arm64'
      // but excluding the main package is often cleaner if it
      // handles platform specifics internally.
    ]
  }
})

