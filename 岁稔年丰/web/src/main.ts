import { createApp } from 'vue';
import { createPinia } from 'pinia';

import App from './App.vue';
import router from './router';

import ElementPlus from 'element-plus';
import locale from 'element-plus/lib/locale/lang/zh-cn';
import 'element-plus/dist/index.css';

import 'default-passive-events' // 添加事件管理者passive，以使页面更加流畅

import '@/styles/element.scss';

const app = createApp(App);

app.use(createPinia());
app.use(ElementPlus,{locale});
app.use(router);

app.mount('#app');
