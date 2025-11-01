const path = require('path');
import { defineConfig } from 'vite';

import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        vue()
    ],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, 'src')
        }
    },

    server: {
        host: '127.0.0.1',
        port: 3030,
        open: false, // 运行是否自动打开浏览器
        // 反向代理解决跨域
        proxy: {
            "/dev": {
                target: 'http://43.138.11.133:8081',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/dev/, "")
            }
        }
    },
    css: {
        preprocessorOptions: {
            scss: {
                additionalData: `@import "./src/styles/common.scss";`,
            }
        }
    }
})
