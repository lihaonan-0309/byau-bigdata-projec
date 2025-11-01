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
    let res: any = await searchCharts({ typeId: 4 });
    data.push(...res.data.list);
    chartName = res.data.name;
}

const initEchart = () => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;

    let xAxisData = data.map((item: any) => item.key);
    let seriesData = data.map((item: any) => item.value);

    option = {
        title: {
            text: chartName,
            subtext: "数据来自于全国农业植物检疫性有害生物分布行政区名录",
        },
        legend: {
            show: true,
            textStyle: {
                color: '#000'
            },
            data: [{
                name: '',
                icon: 'rect'
            }, {
                name: ''
            }]
        },
        tooltip: {
            show: true,
        },
        xAxis: {
            data: xAxisData,
            axisLine: {
                show: true,
                lineStyle: {
                    color: 'rgba(66, 164, 255,0.3)'
                }
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                color: '#000',
                fontSize: 12,
                rotate: -30,
            },
        },
        yAxis: [{
            type: 'value',
            name: "百分比（%）",
            nameTextStyle: {
                color: '#000',
                fontSize: 14
            },
            scale: true,
            axisLine: {
                show: true,
                lineStyle: {
                    color: 'rgba(66, 164, 255,0.4)'
                }
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                color: '#000',
                fontSize: 14
            },
            splitLine: {
                show: true,
                lineStyle: {
                    color: 'rgba(66, 164, 255,0.5)'
                }
            },
        }],
        grid: [
            {
                top: 100,
            }],
        series: [
            {
                name: '防治效果百分比',
                type: 'bar',
                barWidth: 20,
                zlevel: 2,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                        offset: 0,
                        color: '#01B1FF'
                    }, {
                        offset: 1,
                        color: '#033BFF'
                    }], false)

                },
                label: {
                    show: true,
                    fontSize: 13,
                    color: '#14B6F3',
                    position: 'top',
                },
                data: seriesData
            }]
    };

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