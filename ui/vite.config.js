import { defineConfig } from 'vite';

const TARGET = process.env.COUNTER_API || 'http://127.0.0.1:8195';

export default defineConfig({
  server: {
    port: 8196,
    proxy: {
      '/api': {
        target: TARGET,
        changeOrigin: true,
      },
      '/devices': {
        target: TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
});

