<!--
 * @Author: daidai
 * @Date: 2022-02-28 16:16:42
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2022-10-25 09:18:22
 * @FilePath: \web-pc\src\pages\big-screen\view\indexs\left-center.vue
-->
<template>
	<Echart id="leftCenter" :options="options" class="left_center_inner" v-if="pageflag" ref="charts" />
	<Reacquire v-else @onclick="getData" style="line-height:200px">
		重新获取
	</Reacquire>
</template>

<script>
	export default {
		data() {
			return {
				options: {},
				countUserNumData: {
					lockNum: 0,
					onlineNum: 0,
					offlineNum: 0,
					totalNum: 0
				},
				pageflag: true,
				timer: null
			};
		},
		created() {
			this.getData()
		},
		mounted() {},
		beforeDestroy() {
			this.clearData()

		},
		methods: {
			clearData() {
				if (this.timer) {
					clearInterval(this.timer)
					this.timer = null
				}
			},
			getData() {
				this.pageflag = true
				this.init();
			},
			init() {
				this.options = {
					tooltip: {
						trigger: 'axis',
						axisPointer: {
							type: 'cross'
						},
						formatter: function(params) {
							var content = params[0].name + '<br/>'; // 添加横坐标名称
							params.forEach(function(item) {
								content += item.marker + ' ' + item.seriesName + ': ' + item.data +
								'<br/>';

								if (item.dataIndex !== undefined) {
									var landUseValue = item.value[item.dimensionNames.indexOf('land_use')];
									content += 'Land Use: ' + landUseValue + '<br/>';
								}
							});
							return content;
						}
					},
					legend: {
						show: false,
					},
					series: [{
						name: 'Access From',
						type: 'pie',
						label: {
							color: '#7EB7FD', // 设置文字颜色为红色
							formatter: '{b} : {d}%'
						},
						radius: ['40%', '70%'],
						avoidLabelOverlap: false,
						itemStyle: {
							borderRadius: 10,
							borderWidth: 2
						},
						emphasis: {
							label: {
								show: true,
								fontSize: 40,
								fontWeight: 'bold'
							}
						},
						data: [{
								"name": "暗棕壤",
								"value": 77,
								"land_use": ["人工牧草地", "有林地", "旱地", "园地"]
							},
							{
								"name": "黑土",
								"value": 66,
								"land_use": ["旱地"]
							},
							{
								"name": "草甸土",
								"value": 91,
								"land_use": ["其他草地", "天然牧草地", "荒草地", "人工牧草地", "旱地", "灌木林", "滩涂沼泽地", "盐碱地"]
							},
							{
								"name": "沼泽土",
								"value": 22,
								"land_use": ["人工牧草地", "草地,旱地", "旱地", "天然牧草地"]
							},
							{
								"name": "泥炭土",
								"value": 9,
								"land_use": ["其他草地", "天然牧草地", "沼泽地", "旱地", "灌木林"]
							},
							{
								"name": "棕色针叶林土",
								"value": 12,
								"land_use": ["有林地"]
							},
							{
								"name": "水稻土",
								"value": 20,
								"land_use": ["水田"]
							},
							{
								"name": "新积土",
								"value": 8,
								"land_use": ["人工牧草地", "其他林地"]
							},
							{
								"name": "白浆土",
								"value": 23,
								"land_use": ["林地", "天然牧草地", "人工牧草地", "有林地", "旱地"]
							},
							{
								"name": "黑钙土",
								"value": 66,
								"land_use": ["人工牧草地", "旱地"]
							},
							{
								"name": "火山灰土",
								"value": 5,
								"land_use": ["其他草地", "灌木林", "荒地"]
							},
							{
								"name": "风沙土",
								"value": 10,
								"land_use": ["天然牧草地", "旱地"]
							},
							{
								"name": "石质土",
								"value": 2,
								"land_use": ["其他草地", "灌木林"]
							},
							{
								"name": "碱土",
								"value": 16,
								"land_use": ["人工牧草地", "荒草地", "旱地", "盐碱地"]
							}

						]
					}]
				};



			},
		},
	};
</script>
<style lang='scss' scoped>
</style>