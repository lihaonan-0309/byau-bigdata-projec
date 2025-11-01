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
        <div class="textInfo">
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
        <div class="info-box echart-box">
            <div id="echartDom"></div>
            <div class="href">
                <span>百度百科页面链接：<a target="_blank" :href="dataInfo.data.baike_info.baike_url">{{
                    dataInfo.data.baike_info.baike_url
                }}</a></span>
                <span>百度图片页面链接：<a target="_blank" :href="dataInfo.data.baike_info.image_url">{{
                    dataInfo.data.baike_info.baike_url
                }}</a></span>
            </div>
        </div>
    </div>
</template>
<script lang="ts" setup>
import { ElLoading } from 'element-plus'
import { reactive, ref, onMounted } from 'vue';
import type { Ref } from 'vue';
import { uploadBaiduAi } from "@/api/uploadBaiduAi";

import * as echarts from 'echarts';
type EChartsOption = echarts.EChartsOption;

const initEchart = (data: any) => {
    var chartDom = document.getElementById('echartDom')!;
    var myChart = echarts.init(chartDom);
    var option: EChartsOption;
    let xAxisData = data.result.map((item: any) => item.name);
    let seriesData = data.result.map((item: any, i: number) => {
        return {
            "value": item.score,
            itemStyle: {
                color: i === 0 ? '#DC3535' : '#5470C6'
            }
        }
    });

    option = {
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
        baike_info: {
            baike_url: "",
            description: "",
            image_url: "",
        },
        name: "",
        score: 0,
    },
});

const upload = async (e: any) => {
    const data = new FormData();
    data.append("file", e.file);
    openFullScreen();
    let res = await uploadBaiduAi(data, { typeId: 2 });
    dataInfo.data = res.data.result[0];
    setTimeout(() => {
        loading.close();
        initEchart(res.data);
    }, 500)
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

    .textInfo {

        height: 100%;
        width: 36%;

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

            .bottom {
                font-size: 14px;
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


    .echart-box {
        overflow: hidden;
        position: relative;
        margin-bottom: 0 !important;
        width: 60%;
        height: 100%;

        .info-title {
            position: absolute;
            z-index: 10;
            left: 20px;
            top: 20px;
        }

        #echartDom {
            width: 100%;
            height: 80%;
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
</style>
