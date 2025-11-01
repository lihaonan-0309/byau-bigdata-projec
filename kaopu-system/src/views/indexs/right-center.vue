<!--
 * @Author: daidai
 * @Date: 2022-03-01 14:13:04
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2022-09-27 15:04:49
 * @FilePath: \web-pc\src\pages\big-screen\view\indexs\right-top.vue
-->
<template>
	<div  class="right_top_inner">
		<Echart id="rightTop" :options="option" v-if="pageflag" ref="charts" />
	</div>
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

			init() {
				this.option = {
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
						show: false,
					},
					tooltip: {},
					dataset: {
						dimensions: ['name', '氮', '磷', '钾'],
						source: [
							{ name: '玉米', 氮: 158, 磷: 52, 钾: 68 },
							{ name: '水稻', 氮: 162, 磷: 62, 钾: 96 },
							{ name: '小麦', 氮: 165, 磷: 84, 钾: 74 },
							{ name: '高粱', 氮: 271, 磷: 118, 钾: 136 },
							{ name: '小米', 氮: 166, 磷: 145, 钾: 93 },
							{ name: '大豆', 氮: 70, 磷: 89, 钾: 96 },
							{ name: '油菜', 氮: 130, 磷: 74, 钾: 92 },
							{ name: '花生', 氮: 103, 磷: 92, 钾: 125 },
							{ name: '马铃薯', 氮: 172, 磷: 101, 钾: 131 },
							{ name: '甘蔗', 氮: 416, 磷: 230, 钾: 441 }
						]
					},
					xAxis: {
						type: 'category',
					},
					yAxis: {},
					series: [{ type: 'bar' }, { type: 'bar' }, { type: 'bar' }]
				};

			},
		},
	};
</script>
<style lang='scss' scoped>
	.right_top_inner {
		height: 300px;
	}
</style>