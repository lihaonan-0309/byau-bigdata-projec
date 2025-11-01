<template>
    <div class="upload-text" v-if="!dataInfo.data.name">
        <el-empty :image-size="350">
            <el-upload :http-request="upload" class="upload" multiple :show-file-list="false">
                <el-button type="primary">点击上传图片</el-button>
                <template #tip>
                    <div class="el-upload__tip">
                        小于2M的jpg/png文件
                    </div>
                </template>
            </el-upload>
        </el-empty>
    </div>
    <div class="requestInfo" v-show="dataInfo.data.name">
        <div class="textInfo" ref="textInfo">
            <el-card class="box-card">
                <template #header>
                    <div class="card-header">
                        <p class="name">识别结果：<span>{{ dataInfo.data.name }}</span></p>
                        <img :src="dataInfo.data.baike_info.image_url">
                    </div>
                </template>
                <div style="padding: 14px">
                    <span>详情介绍：</span>
                    <div class="bottom">
                        <p>{{ dataInfo.data.baike_info.description }}</p>
                    </div>
                </div>
            </el-card>
            <el-upload :http-request="upload" class="reUpload" multiple :show-file-list="false">
                <div style="color: #fff;">
                    重新上传识别
                </div>
            </el-upload>
        </div>
        <el-tabs v-model="activeName" class="tabs" @tab-click="handleClick" :stretch="true">
            <el-tab-pane class="el-tab-pane" label="识别对象" name="info">
                <div class="info-box echart-box" :style="{ height: tabBodyHeight + 'px' }">
                    <div id="echartDom"></div>
                </div>
            </el-tab-pane>
            <el-tab-pane label="数据统计" name="chart">
                <div class="info-box echart-box" :style="{ height: tabBodyHeight + 'px' }">
                    <div id="echartDom1"></div>
                    <div id="echartDom2"></div>
                    <div id="echartDom3"></div>
                </div>
            </el-tab-pane>
            <el-tab-pane label="防范措施" name="measure">
                <el-card class="box-card" :style="{ height: tabBodyHeight + 'px' }">
                    <template #header>
                        <div class="card-header measure_title">
                            <p class="name">{{ dataInfo.data.measureIntroduce }}</p>
                        </div>
                    </template>
                    <div style="padding: 14px" v-for="(e, i) in dataInfo.data.measure" :key="i">
                        <span class="card_bottom_title">{{ e.title }}</span>
                        <div class="bottom">
                            <p>{{ e.text }}</p>
                        </div>
                    </div>
                </el-card>
            </el-tab-pane>
        </el-tabs>

    </div>
</template>
<script lang="ts" setup>
import { ElLoading } from 'element-plus'
import { reactive, ref, onMounted, nextTick } from 'vue';
import type { Ref } from 'vue';
import { uploadBaiduAi } from "@/api/uploadBaiduAi";
import type { TabsPaneContext } from 'element-plus'
import * as echarts from 'echarts';
type EChartsOption = echarts.EChartsOption;

const nameData: any = {
    "1": "二化螟虫卵",
    "2": "二化螟虫",
    "3": "拟小黄卷叶蛾",
    "4": "草地螟",
    "5": "水稻稻绿蝽",
    "6": "麻皮蝽",
    "7": "水稻稻绿蝽",
    "8": "蝗虫",
}

const initData = [
    {
        "type": "4",
        "name": "草地螟",
        "baike_info": {
            "baike_url": "",
            "image_url": "https://tse1-mm.cn.bing.net/th/id/OIP-C.JXrM6wj5ZVw4vEhjJaDb3gAAAA?w=274&h=168&c=7&r=0&o=5&dpr=1.5&pid=1.7",
            "description": "草地螟Loxostege sticticalis Linnaeus属鳞翅目，夜蛾科。国内分布区北起黑龙江、内蒙古、新疆，南限未过淮河，最南采地江苏、河南、陕西、甘肃、青海，东接前苏联东境、朝鲜北境并滨渤海，西抵新疆、西藏。在各分布区内，常呈间歇性大发生。一种间歇性暴发成灾的害虫，幼虫常常是吃光一块地后，集体迁移至另一块地。可为害小麦、燕麦、玉米、高粱、甜菜、甘蓝、大豆、豌豆、扁豆、马铃薯、向日葵、亚麻、瓜类、胡萝卜、葱、洋葱、茴香、藜科、蓼科、菊科杂草等。"
        },
        measureIntroduce: "草地螟的发生和为害程度与气候条件、虫源基数、田间杂草情况关系密切，因此应因地制宜，采取综合防治措施，制定切实可行的防治对策。",
        measure: [
            {
                title: "诱捕成虫",
                text: "利用成虫趋光性，在成虫发生期，有条件的地区可及时在田间架设频振式杀虫灯或黑光灯或控黑、绿双管灯等进行诱杀。高压汞灯或黑光灯等其它诱虫灯进行成虫诱杀，以达到“杀母抑子”的作用。"
            },
            {
                title: "清除田间杂草、挖隔离带",
                text: "加快旱田铲趟进度，除净大草控制草荒，减少田间落卵量，减少早期孵化幼虫的食料，以降低幼虫密度，减轻危害。在农牧混交区，可提前在农田周围挖沟和打防虫带，以阻止草地螟幼虫大量迁入农田。"
            },
            {
                title: "耕翻土地等农业措施",
                text: "在草地螟集中越冬场所，采取秋翻、春耕办法，通过机械杀伤和土块压伤越冬害虫，增加越冬幼虫死亡率，减轻来年危害程度。通过中耕培土、灌水等农业措施，使幼虫大量死亡，可减轻害虫发生程度。"
            },
            {
                title: "生物防治",
                text: "据报道草地螟的寄生蜂和寄生蝇有70余种，生产上可采用赤眼蜂灭卵，在成虫产卵盛期，每隔5～6d放蜂1次，共2～3次，放蜂量5～30头／h，防治效果可达70％～80％。"
            }
        ],
        option: [
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                title: {
                    text: '病虫高发季节段'
                },
                xAxis: {
                    type: 'category',
                    data: ['春季', '夏季', '秋季', '冬季']
                },
                yAxis: {
                    type: 'value'
                },
                series: [
                    {
                        data: [50, 100, 120, 30],
                        type: 'line'
                    }
                ]
            },
            {
                title: {
                    text: '主要危害作物'
                },
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                legend: {
                    top: '5%',
                    left: 'center'
                },
                series: [
                    {
                        type: 'pie',
                        radius: ['40%', '70%'],
                        avoidLabelOverlap: false,
                        itemStyle: {
                            borderRadius: 10,
                            borderColor: '#fff',
                            borderWidth: 2
                        },
                        label: {
                            show: false,
                            position: 'center'
                        },
                        emphasis: {
                            label: {
                                show: true,
                                fontSize: 40,
                                fontWeight: 'bold'
                            }
                        },
                        labelLine: {
                            show: false
                        },
                        data: [
                            { value: 21.2, name: '甜菜' },
                            { value: 12.5, name: '大豆' },
                            { value: 6.5, name: '向日葵' },
                            { value: 15.1, name: '马铃薯' },
                            { value: 13.0, name: '麻类' },
                            { value: 12.7, name: '蔬菜' },
                            { value: 10.2, name: '药材' },
                            { value: 8.8, name: '其他' }
                        ]
                    }
                ]
            },
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                title: {
                    text: '防治效果'
                },
                xAxis: {
                    type: 'category',
                    data: ['诱捕成虫', '成人灯光诱杀技术', '耕翻土地等农业措施', '生物防治']
                },
                yAxis: {
                    type: 'value'
                },
                series: [
                    {
                        data: [30, 50, 40, 80],
                        type: 'bar'
                    }
                ]
            }
        ]
    },
    {
        "type": "4",
        "name": "草地螟",
        "baike_info": {
            "baike_url": "",
            "image_url": "https://tse1-mm.cn.bing.net/th/id/OIP-C.JXrM6wj5ZVw4vEhjJaDb3gAAAA?w=274&h=168&c=7&r=0&o=5&dpr=1.5&pid=1.7",
            "description": "草地螟Loxostege sticticalis Linnaeus属鳞翅目，夜蛾科。国内分布区北起黑龙江、内蒙古、新疆，南限未过淮河，最南采地江苏、河南、陕西、甘肃、青海，东接前苏联东境、朝鲜北境并滨渤海，西抵新疆、西藏。在各分布区内，常呈间歇性大发生。一种间歇性暴发成灾的害虫，幼虫常常是吃光一块地后，集体迁移至另一块地。可为害小麦、燕麦、玉米、高粱、甜菜、甘蓝、大豆、豌豆、扁豆、马铃薯、向日葵、亚麻、瓜类、胡萝卜、葱、洋葱、茴香、藜科、蓼科、菊科杂草等。"
        },
        "measureIntroduce": "草地螟的发生和为害程度与气候条件、虫源基数、田间杂草情况关系密切，因此应因地制宜，采取综合防治措施，制定切实可行的防治对策。",
        "measure": [
            {
                "title": "诱捕成虫",
                "text": "利用成虫趋光性，在成虫发生期，有条件的地区可及时在田间架设频振式杀虫灯或黑光灯或控黑、绿双管灯等进行诱杀。高压汞灯或黑光灯等其它诱虫灯进行成虫诱杀，以达到“杀母抑子”的作用。"
            },
            {
                "title": "清除田间杂草、挖隔离带",
                "text": "加快旱田铲趟进度，除净大草控制草荒，减少田间落卵量，减少早期孵化幼虫的食料，以降低幼虫密度，减轻危害。在农牧混交区，可提前在农田周围挖沟和打防虫带，以阻止草地螟幼虫大量迁入农田。"
            },
            {
                "title": "耕翻土地等农业措施",
                "text": "在草地螟集中越冬场所，采取秋翻、春耕办法，通过机械杀伤和土块压伤越冬害虫，增加越冬幼虫死亡率，减轻来年危害程度。通过中耕培土、灌水等农业措施，使幼虫大量死亡，可减轻害虫发生程度。"
            },
            {
                "title": "生物防治",
                "text": "据报道草地螟的寄生蜂和寄生蝇有70余种，生产上可采用赤眼蜂灭卵，在成虫产卵盛期，每隔5～6d放蜂1次，共2～3次，放蜂量5～30头／h，防治效果可达70％～80％。"
            }
        ],
        "option": [
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "title": {
                    "text": "病虫高发季节段"
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "春季",
                        "夏季",
                        "秋季",
                        "冬季"
                    ]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [
                            50,
                            100,
                            120,
                            30
                        ],
                        "type": "line"
                    }
                ]
            },
            {
                "title": {
                    "text": "主要危害作物"
                },
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "legend": {
                    "top": "5%",
                    "left": "center"
                },
                "series": [
                    {
                        "name": "Access From",
                        "type": "pie",
                        "radius": [
                            "40%",
                            "70%"
                        ],
                        "avoidLabelOverlap": false,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "#fff",
                            "borderWidth": 2
                        },
                        "label": {
                            "show": false,
                            "position": "center"
                        },
                        "emphasis": {
                            "label": {
                                "show": true,
                                "fontSize": 40,
                                "fontWeight": "bold"
                            }
                        },
                        "labelLine": {
                            "show": false
                        },
                        "data": [
                            {
                                "value": 32,
                                "name": "甜菜"
                            },
                            {
                                "value": 26,
                                "name": "大豆"
                            },
                            {
                                "value": 15,
                                "name": "向日葵"
                            },
                            {
                                "value": 22,
                                "name": "马铃薯"
                            },
                            {
                                "value": 5,
                                "name": "其他"
                            }
                        ]
                    }
                ]
            },
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "title": {
                    "text": "防治效果"
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "诱捕成虫",
                        "成人灯光诱杀技术",
                        "耕翻土地等农业措施",
                        "生物防治"
                    ]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [
                            30,
                            50,
                            40,
                            80
                        ],
                        "type": "bar"
                    }
                ]
            }
        ]
    },
    {
        "type": 7,
        "name": "水稻稻绿蝽",
        "baike_info": {
            "baike_url": "http://baike.baidu.com/item/%E8%92%B2%E8%91%B5/911887",
            "image_url": "https://tse4-mm.cn.bing.net/th/id/OIP-C.cA3rIUkdw_b7LUN6ReWKBwHaFj?w=193&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7",
            "description": "昆虫名，为半翅目，蝽科。中国甜橘产区均有发生。除了危害柑橘外，还为害水稻、玉米、花生、棉花、豆类、十字花科蔬菜、油菜、芝麻、茄子、辣椒、马铃薯、桃、李、梨、苹果等。以成虫、若虫为害烟株，刺吸顶部嫩叶、嫩茎等汁液，常在叶片被刺吸部位先出现水渍状萎蔫，随后干枯。严重时上部叶片或烟株顶梢萎蔫。"
        },
        "measureIntroduce": "",
        "measure": [
            {
                "title": "农业防治",
                "text": "同一作物集中连片种植，避免混栽套种。避免双季稻和中稻插花种植。(2)药剂防治。药剂防治适期在2、3龄若虫盛期，对达到防治指标(水稻百蔸虫量8.7～12.5头)，且水稻离收获期1个月以上、虫口密度较大的田块，可用2.5％溴氰菊酯乳油2 000倍，或20％氰戊菊酯乳油2 000倍液，或2.5％功夫菊酯乳油2 000倍液，或10％吡虫啉可湿性粉剂1 500倍液，或90％敌百虫晶体600～800倍液喷雾。"
            },
            {
                "title": "药剂防治",
                "text": "药剂防治适期在2、3龄若虫盛期，对达到防治指标(水稻百蔸虫量8.7～12.5头)，且水稻离收获期1个月以上、虫口密度较大的田块，可用2.5％溴氰菊酯乳油2 000倍，或20％氰戊菊酯乳油2 000倍液，或2.5％功夫菊酯乳油2 000倍液，或10％吡虫啉可湿性粉剂1 500倍液，或90％敌百虫晶体600～800倍液喷雾。"
            },
            {
                "title": "人工捕杀",
                "text": "利用成虫在早晨和傍晚飞翔活动能力差的特点，进行人工捕杀。"
            }
        ],
        "option": [
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "title": {
                    "text": "病虫高发季节段"
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "春季",
                        "夏季",
                        "秋季",
                        "冬季"
                    ]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [
                            40,
                            100,
                            120,
                            20
                        ],
                        "type": "line"
                    }
                ]
            },
            {
                "title": {
                    "text": "主要危害作物"
                },
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "legend": {
                    "top": "5%",
                    "left": "center"
                },
                "series": [
                    {
                        "type": "pie",
                        "radius": [
                            "40%",
                            "70%"
                        ],
                        "avoidLabelOverlap": false,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "#fff",
                            "borderWidth": 2
                        },
                        "label": {
                            "show": false,
                            "position": "center"
                        },
                        "emphasis": {
                            "label": {
                                "show": true,
                                "fontSize": 40,
                                "fontWeight": "bold"
                            }
                        },
                        "labelLine": {
                            "show": false
                        },
                        "data": [
                            {
                                "value": 39,
                                "name": "水稻"
                            },
                            {
                                "value": 21,
                                "name": "玉米"
                            },
                            {
                                "value": 16,
                                "name": "大豆"
                            },
                            {
                                "value": 10,
                                "name": "小麦"
                            },
                            {
                                "value": 8,
                                "name": "菜豆"
                            },
                            {
                                "value": 8,
                                "name": "其他"
                            }
                        ]
                    }
                ]
            },
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "title": {
                    "text": "防治效果"
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "农业防治",
                        "药剂防治",
                        "人工捕杀"
                    ]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [
                            40,
                            60,
                            20
                        ],
                        "type": "bar"
                    }
                ]
            }
        ]
    },
    {
        "type": 8,
        "name": "蝗虫",
        "baike_info": {
            "baike_url": "http://baike.baidu.com/item/%E8%92%B2%E8%91%B5/911887",
            "image_url": "https://tse2-mm.cn.bing.net/th/id/OIP-C._aV0UIjteW6KPc-GwIPlQQHaE7?w=286&h=190&c=7&r=0&o=5&dpr=1.5&pid=1.7",
            "description": "蝗虫，俗称“蚂蚱”，属直翅目，包括蚱总科（Tetrigoidea）、蜢总科（Eumastacoidea）、蝗总科（Locustoidea）的种类，全世界有超过10000种，我国有1000余种，分布于全世界的热带、温带的草地和沙漠地区。蝗虫主要包括飞蝗和土蝗。在我国飞蝗有东亚飞蝗（Locusta migratoria manilensis (Meyen)）、亚洲飞蝗（Locusta migratoria migratoria (Linnaeus)）和西藏飞蝗（Locusta migratoria tibitensis Chen）3种，其中东亚飞蝗在我国分布范围最广，危害最严重，是造成我国蝗灾的最主要飞蝗种类，主要危害禾本科植物，是农业害虫。"
        },
        "measureIntroduce": "",
        "measure": [
            {
                "title": "农业防治",
                "text": "①、削减蝗虫的食物源 a、蝗虫的食物源主要有：玉米、小麦、高粱、水稻、谷子等，但是不会吃大豆、苜蓿、果树等 b、既然知道蝗虫不会吃大豆、苜蓿、果树及其他林木，那么可以在蝗虫发作地多种植蝗虫不会吃的作物。②、削减蝗虫的生计地 a、首先我们的要了解蝗虫发作地的一些情况，之后在根据具体情况进行防治。 b、要知道很多发作地大部分都是处于地形低的地区，因此可以将地形低的地区改为池塘，用于养鱼或者养虾，这样有利于削减蝗虫的生计，从而更有效的防治蝗虫。 ③、削减蝗虫的产卵地 a、根据蝗虫不同品种的产卵习惯进行削减，比如说东亚飞蝗，就喜欢枯燥暴露的地块产卵 b、可以在枯燥暴露多进行植树造林，增加植物的数量，增加植物掩盖度到达70%以上，蝗虫找不到合适的地块产卵，自然也就能减轻蝗虫带来的损害。"
            },
            {
                "title": "生物防治",
                "text": "①、利用蝗虫的天敌，减少蝗虫数量 蝗虫常见的天敌有青蛙、蜥蜴、鸟、捕食性的甲虫、寄生性的蜂类、寄生蝇类等，因此可以多多培育或者维护蝗虫天敌，这样也是一种很不错的防治方法。 ②、生物农药防治 a、可以选择生物农药防治蝗虫，从而减少蝗虫数量； b、防治蝗虫常用的生物农药有蝗虫微孢子虫、绿僵菌和印楝素等。 ③、利用家禽防治蝗虫 a、利用牧鸡和牧鸭防治蝗虫； b、利用鸡和鸭捕食蝗虫，既能为鸡和鸭提供天然饲料，又可以减少蝗虫数量，添加农业生产效益。"
            }
        ],
        "option": [
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "title": {
                    "text": "病虫高发季节段"
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "春季",
                        "夏季",
                        "秋季",
                        "冬季"
                    ]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [
                            40,
                            120,
                            80,
                            20
                        ],
                        "type": "line"
                    }
                ]
            },
            {
                "title": {
                    "text": "主要危害作物"
                },
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "legend": {
                    "top": "5%",
                    "left": "center"
                },
                "series": [
                    {
                        "type": "pie",
                        "radius": [
                            "40%",
                            "70%"
                        ],
                        "avoidLabelOverlap": false,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "#fff",
                            "borderWidth": 2
                        },
                        "label": {
                            "show": false,
                            "position": "center"
                        },
                        "emphasis": {
                            "label": {
                                "show": true,
                                "fontSize": 40,
                                "fontWeight": "bold"
                            }
                        },
                        "labelLine": {
                            "show": false
                        },
                        "data": [
                            { "value": 28, "name": "棉花" },
                            { "value": 26, "name": "高粱" },
                            { "value": 24, "name": "甘蔗" },
                            { "value": 23, "name": "水稻" },
                            { "value": 21, "name": "花生" }
                        ]
                    }
                ]
            },
            {
                tooltip: {
                    trigger: "item",
                    formatter: '{c}%'
                },
                "title": {
                    "text": "防治效果"
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "农业防治",
                        "生物防治"
                    ]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [
                            60,
                            80
                        ],
                        "type": "bar"
                    }
                ]
            }
        ]
    }
]

const initEchart = (data: any) => {
    let chartDom = document.getElementById('echartDom')!;
    let myChart = echarts.init(chartDom);
    let option: EChartsOption;
    let xAxisData = data.results.map((item: any) => item.name);
    let seriesData = data.results.map((item: any, i: number) => {
        return {
            "value": item.score,
            itemStyle: {
                color: i === 0 ? '#DC3535' : '#5470C6'
            }
        }
    });


    option = {
        tooltip: {
            trigger: "item",
        },
        title: {
            text: "相似度占比",
            left: 'center'
        },
        xAxis: {
            type: 'category',
            data: xAxisData
        },
        yAxis: {
            type: 'value'
        },
        series: [
            {
                data: seriesData,
                type: 'bar'
            }
        ]
    };

    option && myChart.setOption(option);
    window.onresize = function () {//自适应大小
        myChart.resize();
    };
}

const initEchartOther = (optionItem: any, i: any) => {
    let chartDom = document.getElementById(`echartDom${i}`)!;
    let myChart = echarts.init(chartDom);
    let option: EChartsOption = optionItem;
    option && myChart.setOption(option);
    window.onresize = function () {//自适应大小
        myChart.resize();
    };
}


// loading
let loading: any;
const openFullScreen = () => {
    loading = ElLoading.service({
        lock: true,
        text: 'ai识别中',
        background: 'rgba(0, 0, 0, 0.7)',
    })
}

const dataInfo = reactive({
    data: {
        name: "",
        score: 0,
        baike_info: {
            description: "",
            image_url: "",
            baike_url: ""
        },
        option: [],
        measureIntroduce: "",
        measure: [] as any[],
    },
});

const textInfo = ref();
const tabBodyHeight = ref(0);

const dataFormat: any = (type: string) => {
    return initData.find(e => e.type == type)
}

const upload = async (e: any) => {
    const data = new FormData();
    data.append("file", e.file);
    openFullScreen();
    let res = await uploadBaiduAi(data, { typeId: 1 });
    Object.assign(dataInfo.data, dataFormat(res.data.results[0].name))

    res.data.results = res.data.results.map((e: any, i: number) => {
        return {
            score: e.score,
            name: nameData[e.name]
        }
    })

    // 根据左侧栏的高度设置echartdom高度
    let textInfoHeight: any;
    nextTick(() => {
        textInfoHeight = textInfo.value.offsetHeight - 2;
        tabBodyHeight.value = textInfoHeight - 75;
    })

    activeName.value = "info"

    setTimeout(() => {
        loading.close();
        initEchart(res.data);
    }, 1000)
}

// tab切换模块
const activeName = ref('info');
const handleClick = (tab: TabsPaneContext, event: Event) => {
    setTimeout(() => {
        dataInfo.data.option.forEach((e, i) => {
            initEchartOther(e, i + 1);
        });
    }, 100)
}

onMounted(async () => {
})

</script>
<style lang="scss" scoped>
.skeleton {
    display: flex;
    justify-content: space-between;
}

.requestInfo {
    display: flex;
    justify-content: space-between;
    height: 100%;

    .box-card {
        overflow-y: auto;
        height: 95%;

        .bottom {
            font-size: 14px;
        }
    }


    .textInfo {

        height: 100%;
        width: 32%;

        .box-card {
            overflow-y: auto;
            height: 95%;

            .name {
                span {
                    color: #ff0000;
                    font-size: 30px;
                    font-weight: bold;
                }

                margin-bottom: 10px;
            }




            .card-header {
                img {
                    width: 100%;
                    height: auto;
                }
            }
        }

        .reUpload {
            width: 100%;
            height: 5%;
            text-align: center;
            height: 30px;
            line-height: 30px;
            background-color: #7EBD38;
            cursor: pointer;
            border-radius: 0 0 8px 8px;

            div {
                color: #fff !important;
            }
        }
    }

    .tabs {
        width: 64%;
        height: 100%;

        .echart-box {
            overflow: hidden;
            margin-bottom: 0 !important;
            width: 100%;
            height: 100%;
            display: flex;
            flex-wrap: wrap;

            #echartDom {
                width: 100%;
                height: 100%;
            }

            #echartDom1 {
                width: 50%;
                height: 60%;
            }

            #echartDom2 {
                width: 50%;
                height: 60%;
            }

            #echartDom3 {
                width: 100%;
                height: 40%;
            }
        }

        .href {
            margin-left: 50px;

            a {
                display: block;
                margin-bottom: 8px;
            }
        }
    }

}

.measure_title {
    font-size: 18px;
    font-weight: bold;
}
.card_bottom_title{
    font-weight: bold;
}
</style>
