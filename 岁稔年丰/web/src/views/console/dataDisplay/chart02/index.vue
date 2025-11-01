<template>
    <div class="info-box echart-box">
        <div id="echartDom"></div>
    </div>
</template>

<script lang="ts" setup>
import { searchCharts } from "@/api/searchCharts";
import { onMounted } from 'vue';
import * as echarts from 'echarts';
import china from '@/static/china.json';
import "echarts-gl"; //3D地图插件
type EChartsOption = echarts.EChartsOption;

let data: Array<object> = [];
let chartName: string = "";
const getCharts = async () => {
    let res: any = await searchCharts({ typeId: 3 });
    data.push(...res.data.list);
    chartName = res.data.name;
}
//@ts-ignore
echarts.registerMap("china", china);


const initEchart = () => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;

    let xAxisData = data.map((item: any) => item.key);
    let seriesData = data.map((item: any) => item.value);

    option = {
        title: {
            text: chartName,
            textStyle: {
                color: "#000"
            },
            subtextStyle: {
                color: "#000"
            }
        },
        // backgroundColor: 'rgba(0,0,0,1)',
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'none'
            },
            formatter: function (params: any) {
                return params[0].name + " " + params[0].value + "种";
            }
        },
        yAxis: [
            {
                inverse: true,
                type: 'category',
                data: xAxisData,
                // offset: 40,
                axisTick: {
                    show: false
                },
                axisLine: {
                    show: false
                },
                axisLabel: {
                    color: '#000'
                }
            }

        ],
        xAxis: [
            {
                type: 'value',
                name: "种",
                nameTextStyle: {
                    color: "#000",
                    fontSize: 12,
                },
                axisLine: {
                    lineStyle: {
                        color: '#0a3e98'
                    }
                },
                axisTick: {
                    show: false,
                },
                axisLabel: {
                    formatter: '{value}',
                    color: "#000",
                },
            }
        ],
        // grid: [
        //     {
        //         width: "25%",
        //         right: "10%",
        //     }],
        series: [{
            type: 'bar',
            barWidth: 15,
            itemStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                    offset: 0,
                    color: "rgba(254,215,46,1)",
                },
                {
                    offset: 1,
                    color: "rgba(254,215,46,0)",
                },
                ]),
                borderRadius: [0, 30, 30, 0] //圆角大小
            },
            data: seriesData
        }]
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