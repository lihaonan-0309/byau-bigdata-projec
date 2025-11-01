import Layout from '../layout/index.vue';
//控制台页面

export default [

    // {
    //     // path: '/',
    //     path: '/overview',
    //     name: "Layout",
    //     component: Layout,
    //     redirect: '/overview',
    //     meta: {
    //         title: '总览',
    //         icon: '/src/assets/navIcons/nav_overview.png',
    //     },
    //     children: [
    //         {
    //             path: '/overview',
    //             name: "overview",
    //             component: () => import('../views/console/Overview/index.vue'),
    //             meta: {
    //                 title: '首页',
    //                 icon: '/src/assets/navIcons/nav_overview.png',
    //                 activeIcon: '/src/assets/navIcons/nav_overview_active.png'
    //             }

    //         }
    //     ]
    // },


    {
        path: '/consoleHome',
        name: "Layout",
        component: Layout,
        redirect: '/consoleHome',
        meta: {
            title: '首页',
            icon: '/src/assets/navIcons/nav_01.png',
        },
        children: [
            {
                path: '/consoleHome',
                name: "consoleHome",
                component: () => import('../views/console/consoleHome/index.vue'),
                meta: {
                    title: '首页',
                    icon: '/src/assets/navIcons/nav_overview.png',
                    activeIcon: '/src/assets/navIcons/nav_overview_active.png'
                }

            }
        ]
    },
    {
        path: '/aIRecognition',
        name: "aIRecognition",
        component: Layout,
        redirect: '/aIRecognition',
        meta: {
            title: 'AI识别',
            icon: '/src/assets/navIcons/nav_01.png',
        },
        children: [
            {
                path: '/algorithm01',
                name: "algorithm01",
                component: () => import('../views/console/aIRecognition/algorithm01/index.vue'),
                meta: {
                    title: '病虫害识别',
                }

            },
            {
                path: '/algorithm02',
                name: "algorithm02",
                component: () => import('../views/console/aIRecognition/algorithm02/index.vue'),
                meta: {
                    title: '农作物识别',
                }

            }
        ]
    },
    {
        path: '/dataDisplay',
        name: "dataDisplay",
        component: Layout,
        redirect: '/dataDisplay',
        meta: {
            title: '数据统计',
            icon: '/src/assets/navIcons/nav_02.png',
            activeIcon: '/src/assets/navIcons/nav_02_active.png',
        },
        children: [
            {
                path: '/chart05',
                name: "chart05",
                component: () => import('../views/console/dataDisplay/chart05/index.vue'),
                meta: {
                    title: '农作物八大类害虫汇总',
                }
            },
            {
                path: '/chart04',
                name: "chart04",
                component: () => import('../views/console/dataDisplay/chart04/index.vue'),
                meta: {
                    title: '各省农作物病虫害',
                }
            },
            {
                path: '/chart01',
                name: "chart01",
                component: () => import('../views/console/dataDisplay/chart01/index.vue'),
                meta: {
                    title: '农业生产病虫害经济损失',
                }
            },
            {
                path: '/chart03',
                name: "chart03",
                component: () => import('../views/console/dataDisplay/chart03/index.vue'),
                meta: {
                    title: '病虫害防治效果排行',
                }
            },
            {
                path: '/chart07',
                name: "chart07",
                component: () => import('../views/console/dataDisplay/chart07/index.vue'),
                meta: {
                    title: '主要农产品产量',
                }
            },
            {
                path: '/chart06',
                name: "chart06",
                component: () => import('../views/console/dataDisplay/chart06/index.vue'),
                meta: {
                    title: '常见农作物六年产量更替图',
                }
            },
            {
                path: '/chart08',
                name: "chart08",
                component: () => import('../views/console/dataDisplay/chart08/index.vue'),
                meta: {
                    title: '各省水稻播种面积亩产量',
                }
            },
            {
                path: '/chart09',
                name: "chart09",
                component: () => import('../views/console/dataDisplay/chart09/index.vue'),
                meta: {
                    title: '2022年世界大米产量排行',
                }
            },
            {
                path: '/chart11',
                name: "chart11",
                component: () => import('../views/console/dataDisplay/chart11/index.vue'),
                meta: {
                    title: '中国大米产量和需求量',
                }
            },
            {
                path: '/chart10',
                name: "chart10",
                component: () => import('../views/console/dataDisplay/chart10/index.vue'),
                meta: {
                    title: '2012~2021玉米播种面积',
                }
            }
        ]
    }
]



