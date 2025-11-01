<!--
 * @Author: daidai
 * @Date: 2022-02-28 16:16:42
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2022-07-20 17:57:11
 * @FilePath: \web-pc\src\pages\big-screen\view\indexs\left-center.vue
-->
<template>
	<div>
		<!-- <ul class="user_Overview flex" v-if="pageflag">
			<li class="user_Overview-item" style="color: #00fdfa">
				<div class="user_Overview_nums allnum ">
					<dv-digital-flop :config="config" style="width:100%;height:100%;" />
				</div>
				<p>白天温度</p>
			</li>
			<li class="user_Overview-item" style="color: #07f7a8">
				<div class="user_Overview_nums online">
					{{onlineconfig.number[0]}}
				</div>
				<p>天气</p>
			</li>
			<li class="user_Overview-item" style="color: #e3b337">
				<div class="user_Overview_nums offline">
					<dv-digital-flop :config="offlineconfig" style="width:100%;height:100%;" />
				</div>
				<p>夜晚温度</p>
			</li>
		</ul>
		<Reacquire v-else @onclick="getData" line-height="200px">
			重新获取
		</Reacquire> -->

		<div class="today">
			<div>
				今日：{{lastDay[0].dayweather}} - {{lastDay[0].nightweather}}
			</div>
			<div>{{lastDay[0].daytemp}}°~{{lastDay[0].nighttemp}}°</div>
			<div>
				{{lastDay[0].daywind}}风{{lastDay[0].daypower}}级 ~ {{lastDay[0].nightwind}}风{{lastDay[0].nightpower}}级
			</div>
		</div>

		<div class="lastday">
			<div>
				<span>明天：</span>白天：{{lastDay[1].daytemp}}°，夜晚：{{lastDay[1].nighttemp}}°，{{lastDay[1].dayweather}}，{{lastDay[1].daywind}}风{{lastDay[1].daypower}}级
				~ {{lastDay[1].nightwind}}风{{lastDay[1].nightpower}}级
			</div>
			<div>
				<span>后天：</span>白天：{{lastDay[2].daytemp}}°，夜晚：{{lastDay[2].nighttemp}}°，{{lastDay[2].dayweather}}，{{lastDay[2].daywind}}风{{lastDay[2].daypower}}级
				~ {{lastDay[2].nightwind}}风{{lastDay[2].nightpower}}级
			</div>
		</div>
	</div>

</template>

<script>
	import axios from 'axios';
	let style = {
		fontSize: 24
	}
	export default {
		data() {
			return {
				lastDay: [],
				options: {},
				userOverview: {
					alarmNum: 0,
					offlineNum: 0,
					onlineNum: 0,
					totalNum: 0,
				},
				pageflag: true,
				timer: null,
				config: {
					number: [100],
					content: '{nt}',
					style: {
						...style,
						// stroke: "#00fdfa",
						fill: "#00fdfa",
					},
				},
				onlineconfig: {
					number: [0],
					content: '{nt}',
					style: {
						...style,
						// stroke: "#07f7a8",
						fill: "#07f7a8",
					},
				},
				offlineconfig: {
					number: [0],
					content: '{nt}',
					style: {
						...style,
						// stroke: "#e3b337",
						fill: "#e3b337",
					},
				},
				laramnumconfig: {
					number: [0],
					content: '{nt}',
					style: {
						...style,
						// stroke: "#f5023d",
						fill: "#f5023d",
					},
				}

			};
		},
		filters: {
			numsFilter(msg) {
				return msg || 0;
			},
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
			async getData() {
				const data = await axios.get(
					"https://restapi.amap.com/v3/weather/weatherInfo?key=52c86ab73911ae543a72953e50e31eaf&city=230100&extensions=all"
				);

				this.lastDay = data.forecasts[0].casts;

				this.pageflag = true;

				this.onlineconfig = {
					...this.onlineconfig,
					number: [data.forecasts[0].casts[0].dayweather]
				}
				this.config = {
					...this.config,
					number: [Number(data.forecasts[0].casts[0].daytemp)]
				}
				this.offlineconfig = {
					...this.offlineconfig,
					number: [Number(data.forecasts[0].casts[0].nighttemp)]
				}
				// this.laramnumconfig = {
				//     ...this.laramnumconfig,
				//     number: [res.data.alarmNum]
				// }

			},
		},
	};
</script>
<style lang='scss' scoped>
	.user_Overview {
		li {
			flex: 1;

			p {
				text-align: center;
				height: 16px;
				font-size: 16px;
			}

			.user_Overview_nums {
				width: 100px;
				height: 100px;
				text-align: center;
				line-height: 100px;
				font-size: 22px;
				margin: 10px auto 30px;
				background-size: cover;
				background-position: center center;
				position: relative;

				&::before {
					content: '';
					position: absolute;
					width: 100%;
					height: 100%;
					top: 0;
					left: 0;
				}

				&.bgdonghua::before {
					animation: rotating 14s linear infinite;
				}
			}

			.allnum {

				// background-image: url("../../assets/img/left_top_lan.png");
				&::before {
					background-image: url("../../assets/img/left_top_lan.png");

				}
			}

			.online {
				&::before {
					background-image: url("../../assets/img/left_top_lv.png");

				}
			}

			.offline {
				&::before {
					background-image: url("../../assets/img/left_top_huang.png");

				}
			}

			.laramnum {
				&::before {
					background-image: url("../../assets/img/left_top_hong.png");

				}
			}
		}
	}

	.lastday {
		margin-top: 50px;
		padding: 0 30px;
		text-align: center;

		&>div {
			margin-bottom: 10px;
			font-size: 14px;

			span {
				font-weight: bold;
				font-size: 16px;
			}
		}
	}

	.today {
		padding-top: 20px;
		text-align: center;

		&>div {
			margin-bottom: 20px;
		}

		&>div:first-child {
			font-size: 25px;
		}

		&>div:nth-child(2) {
			font-size: 20px;
		}

		&>div:nth-child(2) {
			font-size: 18px;
		}
	}
</style>