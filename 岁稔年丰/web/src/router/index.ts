import { createRouter, createWebHistory } from 'vue-router'
import AiLayout from '../layout/homeLayout.vue';
import console from './console';

export const home = [
    {
        path: '/',
        name: "ai_layout",
        component: AiLayout,
        redirect: '/home',
        children: [
            {
                path: '/home',
                name: 'home',
                component: () => import('../views/home/index.vue'),
                meta: {
                    title: '首页',
                    isHeader: true, //是否为头部menu列表
                }
            },
        ]
    },

]


const routes = [...home, ...console]


export const constantRoutes = routes;

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router
