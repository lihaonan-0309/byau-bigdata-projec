<template>
	<div class="center_bottom">
		<Echart :options="options" id="bottomLeftChart" class="echarts_bottom"></Echart>
	</div>
</template>

<script>
	import axios from 'axios';
	import { GETNOBASE } from 'api'
	import { graphic } from "echarts";
	export default {
		data() {
			return {
				options: {},
			};
		},
		props: {},
		mounted() {
			this.getData();
		},
		methods: {
			getData() {
				this.pageflag = true;
				// axios.get('http://82.156.0.244:59687/200ri')
				//   .then(response => {
				//     // 请求成功处理
				//     console.log(response.data);
				//   })
				GETNOBASE("http://82.156.0.244:59687/200ri").then((res) => {
					this.init(res["200ri"].reverse());

				});
			},
			init(newData) {
				let dateList = [];
				let data1 = [];
				let data2 = [];
				let data3 = [];

				for (let s of newData) {
					dateList.push(s.post_time);
					data1.push(s.clz);
					data2.push(s.ncp);
					data3.push(s.ly);
				}

				this.options = {
					axisLine: {
						lineStyle: {
							color: "rgba(31,99,163,.1)",
						},
					},
					axisLabel: {
						color: "#7EB7FD",
						fontWeight: "500",
					},
					legend: {
						data: ["'菜篮子'产品批发价格200指数", "农产品批发价格200指数", "粮油产品批发价格200指数"]
					},
					xAxis: {
						type: 'category',
						data: dateList
					},
					yAxis: {
						type: 'value',
						min: 115,
					},
					series: [{
							data: data1,
							type: 'line',
							smooth: true
						},
						{
							data: data2,
							type: 'line',
							smooth: true
						},
						{
							data: data3,
							type: 'line',
							smooth: true
						}
					]
				};
			},
		},
	};
</script>
<style lang="scss" scoped>
	.center_bottom {
		width: 100%;
		height: 100%;

		.echarts_bottom {
			width: 100%;
			height: 100%;
		}
	}
</style>