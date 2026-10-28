import {defineConfig} from 'vite';
export default defineConfig({build:{target:'es2022',rollupOptions:{input:{main:'index.html',compare:'compare.html',review:'review.html'}}}});
