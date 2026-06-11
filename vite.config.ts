import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    port: process.env.PORT ? Number(process.env.PORT) : undefined,
  },
  build: {
    rollupOptions: {
      input: {
        home: fileURLToPath(new URL('./index.html', import.meta.url)),
        blackhole: fileURLToPath(new URL('./blackhole/index.html', import.meta.url)),
        tutorial: fileURLToPath(new URL('./blackhole/tutorial.html', import.meta.url)),
        kerr: fileURLToPath(new URL('./kerr/index.html', import.meta.url)),
        kerrTutorial: fileURLToPath(new URL('./kerr/tutorial.html', import.meta.url)),
        fall: fileURLToPath(new URL('./fall/index.html', import.meta.url)),
        fallTutorial: fileURLToPath(new URL('./fall/tutorial.html', import.meta.url)),
        gr: fileURLToPath(new URL('./gr/index.html', import.meta.url)),
      },
    },
  },
})
