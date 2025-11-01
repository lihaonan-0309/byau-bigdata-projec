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
    let res: any = await searchCharts({ typeId: 6 });
    data.push(...res.data.list);
    chartName = res.data.name;
}
const initEchart = () => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;
    let xAxisData = data.map((item: any) => item.key); xAxisData.unshift("product");
    let seriesData1 = data.map((item: any) => item["水稻"]); seriesData1.unshift("水稻");
    let seriesData2 = data.map((item: any) => item["玉米"]); seriesData2.unshift("玉米");
    let seriesData3 = data.map((item: any) => item["小麦"]); seriesData3.unshift("小麦");
    option = {
        title: {
            text: chartName,
            subtext: "数据来自于国家粮食和物资储备局",
        },
        legend: {},
        tooltip: {
            trigger: 'axis',
            // showContent: false
        },
        dataset: {
            source: [xAxisData, seriesData1, seriesData2, seriesData3]
        },
        xAxis: { type: 'category' },
        yAxis: { gridIndex: 0 },
        grid: { top: '55%' },
        series: [
            {
                type: 'line',
                smooth: true,
                seriesLayoutBy: 'row',
                emphasis: { focus: 'series' }
            },
            {
                type: 'line',
                smooth: true,
                seriesLayoutBy: 'row',
                emphasis: { focus: 'series' }
            },
            {
                type: 'line',
                smooth: true,
                seriesLayoutBy: 'row',
                emphasis: { focus: 'series' }
            },
            {
                type: 'pie',
                id: 'pie',
                radius: '30%',
                center: ['50%', '25%'],
                emphasis: {
                    focus: 'self'
                },
                label: {
                    formatter: '{b}: {@2016} ({d}%)'
                },
                encode: {
                    itemName: 'product',
                    value: '2016',
                    tooltip: '2016'
                }
            }
        ]
    };

    myChart.on('updateAxisPointer', function (event: any) {
        const xAxisInfo = event.axesInfo[0];
        if (xAxisInfo) {
            const dimension = xAxisInfo.value + 1;
            myChart.setOption<echarts.EChartsOption>({
                series: {
                    id: 'pie',
                    label: {
                        formatter: '{b}: {@[' + dimension + ']} ({d}%)'
                    },
                    encode: {
                        value: dimension,
                        tooltip: dimension
                    }
                }
            });
        }
    });

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