<!--
 * @Author: daidai
 * @Date: 2022-03-01 14:13:04
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2022-09-27 15:04:49
 * @FilePath: \web-pc\src\pages\big-screen\view\indexs\right-top.vue
-->
<template>
	<Echart id="rightTop" :options="option" class="right_top_inner" v-if="pageflag" ref="charts" />
	<Reacquire v-else @onclick="getData" style="line-height: 200px">
		重新获取
	</Reacquire>
</template>

<script>
	// import { currentGET } from "api/modules";
	import { graphic } from "echarts"
	export default {
		data() {
			return {
				option: {},
				pageflag: false,
				timer: null,
			};
		},
		created() {

		},

		mounted() {
			this.getData();
		},
		beforeDestroy() {
			this.clearData();
		},
		methods: {
			clearData() {
				if (this.timer) {
					clearInterval(this.timer);
					this.timer = null;
				}
			},
			getData() {
				this.pageflag = true;
				this.init();
			},

			init(xData, yData, yData2) {
				let dateList = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008,
					2007, 2006, 2005, 2004
				].reverse();
				let data1 = [5079.2, 5191.26, 5250.65, 5403.59, 5653.42, 5859.41, 5984.41, 6022.6, 5995.94, 5911.86,
					5838.85, 5704.24, 5561.68, 5404.35, 5239.02, 5107.83, 4927.69, 4766.22, 4636.58
				].reverse();
				let data2 = [1654.18, 1745.32, 1833.86, 1930.21, 2065.43, 2221.81, 2310.46, 2361.57, 2392.86, 2394.24,
					2399.89, 2381.42, 2353.68, 2329.88, 2302.88, 2297.21, 2262.5, 2229.29, 2221.89
				].reverse();
				let data3 = [563.19, 627.15, 653.85, 681.58, 728.88, 797.59, 829.99, 843.06, 845.34, 830.61, 828.57,
					819.19, 805.64, 797.73, 780.08, 773.02, 769.51, 743.84, 735.96
				].reverse();
				let data4 = [493.16, 524.75, 541.91, 561.13, 590.28, 619.74, 636.9, 642.28, 641.94, 627.42, 617.71, 605.13,
					586.44, 564.29, 545.2, 533.62, 509.75, 489.45, 467.29
				].reverse();
				let data5 = [2368.68, 2294.04, 2221.02, 2230.67, 2268.84, 2220.27, 2207.07, 2175.69, 2115.81, 2057.48,
					1989.97, 1895.09, 1798.5, 1698.69, 1608.6, 1502.98, 1385.85, 1303.18, 1204
				].reverse();

				this.option = {
					xAxis: {
						type: "category",
						data: dateList,
						boundaryGap: false, // 不留白，从原点开始
						splitLine: {
							show: true,
							lineStyle: {
								color: "rgba(31,99,163,.2)",
							},
						},
						axisLine: {
							// show:false,
							lineStyle: {
								color: "rgba(31,99,163,.1)",
							},
						},
						axisLabel: {
							color: "#7EB7FD",
							fontWeight: "500",
						},
					},
					yAxis: {
						type: "value",
						splitLine: {
							show: true,
							lineStyle: {
								color: "rgba(31,99,163,.2)",
							},
						},
						axisLine: {
							lineStyle: {
								color: "rgba(31,99,163,.1)",
							},
						},
						axisLabel: {
							color: "#7EB7FD",
							fontWeight: "500",
						},
					},
					tooltip: {
						trigger: "axis",
						backgroundColor: "rgba(0,0,0,.6)",
						borderColor: "rgba(147, 235, 248, .8)",
						textStyle: {
							color: "#FFF",
						},
					},
					grid: {
						//布局
						show: true,
						left: "10px",
						right: "30px",
						bottom: "10px",
						top: "28px",
						containLabel: true,
						borderColor: "#1F63A3",
					},
					series: [{
							data: data1,
							type: "line",
							smooth: true,
							symbol: "none", //去除点
							name: "农用化肥施用折纯量",
						},
						{
							data: data2,
							type: "line",
							smooth: true,
							symbol: "none", //去除点
							name: "农用氮肥施用折纯量",
						},

						{
							data: data3,
							type: "line",
							smooth: true,
							symbol: "none", //去除点
							name: "农用磷肥施用折纯量",
						},

						{
							data: data4,
							type: "line",
							smooth: true,
							symbol: "none", //去除点
							name: "农用钾肥施用折纯量",
						},

						{
							data: data5,
							type: "line",
							smooth: true,
							symbol: "none", //去除点
							name: "农用复合肥施用折纯量",
						},




					],
				};
			},
		},
	};
</script>
<style lang='scss' scoped>
	.right_top_inner {
		margin-top: -8px;
	}
</style>