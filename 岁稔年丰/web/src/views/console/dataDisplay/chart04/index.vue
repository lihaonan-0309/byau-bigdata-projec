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
    let res: any = await searchCharts({ typeId: 5 });
    data.push(...res.data.list);
    chartName = res.data.name;
}

const initEchart = () => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;

    let xAxisData = data.map((item: any) => item.key);
    let seriesData1 = data.map((item: any) => item.disease);
    let seriesData2 = data.map((item: any) => item.insectPest);

    option = {
        title: {
            text: chartName,
            subtext: "数据来自于全国农业植物检疫性有害生物分布行政区名录",
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
            data: ["病害", "虫害"],
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
                minInterval: 0,
                max: 5,
                axisLabel: {
                    show: true,
                    color: '#000',
                    padding: 0,
                    fontSize: 16,
                    formatter: function (value:any, index:any) {
                        if (!value) return value;
                        return value + '.0';
                    },
                },
            },

        ],
        series: [
            {
                name: "病害",
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
                name: "虫害",
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