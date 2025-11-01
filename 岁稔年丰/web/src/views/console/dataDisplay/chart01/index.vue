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
    let res: any = await searchCharts({ typeId: 2 });
    data.push(...res.data.list);
    chartName = res.data.name;
}



const initEchart = () => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;

    let xAxisData = data.map((item: any) => item.key);
    let seriesData = data.map((item: any) => item.value);

    let maxSeriesData = []
    const MAX = Math.max(...seriesData)
    for (let i = 0; i < seriesData.length; i++) {
        maxSeriesData.push(MAX)
    }
    let barLinearColors = [
        new echarts.graphic.LinearGradient(0, 1, 1, 1, [
            { offset: 0, color: "#EB3B5A" },
            { offset: 1, color: "#FE9C5A" }
        ]),
        new echarts.graphic.LinearGradient(0, 1, 1, 1, [
            { offset: 0, color: "#FA8231" },
            { offset: 1, color: "#FFD14C" }
        ]),
        new echarts.graphic.LinearGradient(0, 1, 1, 1, [
            { offset: 0, color: "#F7B731" },
            { offset: 1, color: "#FFEE96" }
        ]),
        new echarts.graphic.LinearGradient(0, 1, 1, 1, [
            { offset: 0, color: "#395CFE" },
            { offset: 1, color: "#2EC7CF" }
        ])
    ]

    function rankBarColor(cData: Array<number>) {
        let tempData: Array<Object> = [];

        cData.forEach((item, index) => {
            tempData.push({
                value: item,
                itemStyle: {
                    color: index > 3 ? barLinearColors[3] : barLinearColors[index]
                }
            })
        })
        return tempData;
    }
    option = {
        title: {
            text: chartName,
            subtext: "数据来自于百纳文秘网",
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(50,50,50,.3)',
            textStyle: {
                color: '#222'
            },
            formatter: function (value: any, index) {
                let e = value[0];

                return `${e.name}<br/>经济损失 : ${e.value} 亿元`;
            },
        },
        xAxis: {
            type: "value",
            splitLine: { show: false },
            axisLabel: { show: false },
            axisTick: { show: false },
            axisLine: { show: false }
        },
        yAxis: [
            {
                type: "category",
                inverse: true,
                axisLine: { show: false },
                axisTick: { show: false },
                data: xAxisData,
                axisLabel: {
                    rich: {
                        nt1: {
                            color: "#fff",
                            backgroundColor: '#EB3B5A',
                            width: 16,
                            height: 16,
                            fontSize: 12,
                            align: "center",
                            borderRadius: 100,
                            padding: [1, 0, 0, 0]
                        },
                        nt2: {
                            color: "#fff",
                            backgroundColor: '#FA8231',
                            width: 16,
                            height: 16,
                            fontSize: 12,
                            align: "center",
                            borderRadius: 100,
                            padding: [1, 0, 0, 0]
                        },
                        nt3: {
                            color: "#fff",
                            backgroundColor: '#F7B731',
                            width: 16,
                            height: 16,
                            fontSize: 12,
                            align: "center",
                            borderRadius: 100,
                            padding: [1, 0, 0, 0]
                        },
                        nt: {
                            color: "#fff",
                            backgroundColor: '#00a9c8',
                            width: 16,
                            height: 16,
                            fontSize: 12,
                            align: "center",
                            borderRadius: 100,
                            padding: [1, 0, 0, 0]
                        }
                    },
                    formatter: function (value: number, index: number) {
                        let idx = index + 1
                        if (idx <= 3) {
                            return ["{nt" + idx + "|" + idx + "}"].join("\n");
                        } else {
                            return ["{nt|" + idx + "}"].join("\n");
                        }
                    }
                }
            },
            {//名称
                type: 'category',
                offset: -10,
                position: "left",
                inverse: true,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                    color: '#333',
                    align: "left",
                    verticalAlign: "bottom",
                    lineHeight: 32,
                    fontSize: 12
                },
                data: xAxisData
            },
        ],
        series: [
            {
                zlevel: 1,
                type: "bar",
                barWidth: 16,
                data: rankBarColor(seriesData),
                itemStyle: {
                    borderRadius: 30
                },
                label: {
                    show: true,
                    fontSize: 12,
                    color: "#fff"
                }
            },
            {
                type: "bar",
                barWidth: 16,
                barGap: "-100%",
                itemStyle: {
                    borderRadius: 30,
                    color: 'rgba(0,0,0,0.04)'
                },
                data: maxSeriesData
            }
        ]
    };


    option && myChart.setOption(option);
    window.onresize = function () {//自适应大小
        myChart.resize();
    };
}

onMounted(async () => {
    await getCharts();
    // setTimeout(() => {
    initEchart()
    // }, 1000);
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