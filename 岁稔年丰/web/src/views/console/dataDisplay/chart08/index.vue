<template>
    <div class="info-box echart-box">
        <div id="echartDom"></div>
    </div>
</template>

<script lang="ts" setup>
import { searchCharts } from "@/api/searchCharts";
import { onMounted } from 'vue';
import * as echarts from 'echarts';
type EChartsOption = echarts.EChartsOption;

let data: Array<object> = [];
let chartName: string = "";
const getCharts = async () => {
    let res: any = await searchCharts({ typeId: 9 });
    data.push(...res.data.list);
    chartName = res.data.name;
}

const initEchart = () => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;

    let xAxisData = data.map((item: any) => item.key);
    let seriesData1 = data.map((item: any) => item["播种"]);
    let seriesData2 = data.map((item: any) => item["产量"]);
    let seriesData3 = data.map((item: any) => item["单产"]);

    option = {
        title: {
            text: chartName,
            subtext: "数据来自于国家统计局",
        },
        grid: {
            top: '100'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                crossStyle: {
                    color: '#999'
                },
                lineStyle: {
                    type: 'dashed'
                }
            }
        },
        legend: [{
            show: true,
            // icon: 'rect',
            top: '2%',
            itemWidth: 20,
            itemHeight: 10,
            textStyle: {
                fontSize: 14,
                color: '#000',
            },
            data: ["播种", "产量", "单产"],
        },
        ],
        xAxis: [{
            type: 'category',
            nameTextStyle: {
                color: '#000',
                fontSize: 12,
            },
            boundaryGap: true,
            axisLine: { //坐标轴轴线相关设置。数学上的x轴
                show: true,
                lineStyle: {
                    color: 'rgba(0, 0, 0, 0.2)'
                },
            },
            axisLabel: { //坐标轴刻度标签的相关设置
                interval: 0,
                color: '#000',
                padding: 0,
                fontSize: 12,
            },
            data: xAxisData
        }],
        yAxis: [
            {
                type: 'value',
                name: "单位：千公顷、万吨、公斤/亩",
                nameTextStyle: {
                    color: '#000',
                },
                boundaryGap: false,
                splitLine: {
                    show: true,
                    lineStyle: {
                        color: 'rgba(0, 0, 0, 0.2)'
                    },
                },
                axisLine: {
                    show: false,
                },
                axisLabel: {
                    show: true,
                    color: '#000',
                    padding: 0,
                    fontSize: 16,

                },
            },

        ],
        series: [
            {
                name: "播种",
                type: 'line',
                showAllSymbol: true,
                showSymbol: true,
                symbol: 'triangle',
                symbolSize: 10,
                smooth: true,//平滑
                lineStyle: {
                    width: 2,
                    color: "rgba(248, 181, 81, 1)", // 线条颜色
                    type: 'solid',
                },
                itemStyle: {//标记点样式
                    color: 'rgba(248, 181, 81, 1)',
                    borderWidth: 0,
                    borderColor: "#000"
                },

                tooltip: {
                    show: true
                },
                label: {
                    show: false,
                },
                data: seriesData1
            },
            {
                name: "产量",
                type: 'line',
                showAllSymbol: true,
                showSymbol: true,
                symbol: 'rect',
                symbolRotate: 45,
                symbolSize: 10,
                smooth: true,//平滑
                lineStyle: {
                    width: 2,
                    color: "rgba(0, 163, 255, 1)", // 线条颜色
                    type: 'solid',
                },
                itemStyle: {//标记点样式
                    color: 'rgba(0, 163, 255, 1)',
                    borderWidth: 0,
                    borderColor: "#000"
                },

                tooltip: {
                    show: true
                },
                label: {
                    show: false,
                },
                data: seriesData2
            },
            {
                name: '单产',
                yAxisIndex: 0,
                type: 'bar',
                itemStyle: {
                    color: {
                        type: 'linear',
                        x: 0,
                        y: -0.2,
                        x2: 0,
                        y2: 1,
                        colorStops: [
                            {
                                offset: 0,
                                color: 'rgba(108,224,158, 1)'
                            },
                            {
                                offset: 1,
                                color: 'rgba(108,224,158, 0.13)'
                            }
                        ]
                    }
                },
                data: seriesData3
            },
        ]
    }


    option && myChart.setOption(option);
    window.onresize = function () {//自适应大小
        myChart.resize();
    };
}

onMounted(async () => {
    await getCharts();
    initEchart()
})

</script>

<style lang="scss" scoped>
.echart-box {
    overflow: hidden;
    position: relative;
    margin-bottom: 0 !important;
    height: 100%;

    .info-title {
        position: absolute;
        z-index: 10;
        left: 20px;
        top: 20px;
    }

    #echartDom {
        width: 100%;
        height: 100%;
    }
}
</style>