<template>
	<div class="box">
		<div class="content">
			<div class="content_top">
				<dv-border-box-13>
					<div class="sub_box">
						<div class="title">
							<el-select v-model="yuchuli" placeholder="请选择">
								<el-option label="平滑处理" value="smooth">
								</el-option>
								<el-option label="一阶导数处理" value="derive">
								</el-option>
								<el-option label="多元散射校正处理" value="msc">
								</el-option>
								<el-option label="标准正态变换处理" value="standardize">
								</el-option>
								<el-option label="标准正态变换+去趋势处理" value="detrend">
								</el-option>
							</el-select>
							<el-upload class="upload-demo" :action="'http://82.156.0.244:59687/'+yuchuli"
								:multiple="false" :file-list="fileList" :before-upload="yuchuliBefore"
								:on-success="yuchuliSuccess" multiple :limit="1">
								<el-button style="margin-left: 10px;" size="small" type="primary">点击上传
									csv、excel文件</el-button>
							</el-upload>
						</div>
						<div class="echartsBox">
							<div>
								<dv-loading v-if="loading">Loading...</dv-loading>
								<Echart id="bbb" :options="yuchuliOptionbbb" v-if="yuchuliShow" ref="chartsbbb" />
							</div>
							<div style="position: relative;">
								<dv-loading v-if="loading">Loading...</dv-loading>
								<Echart id="aaa" :options="yuchuliOptionaaa" v-if="yuchuliShow" ref="chartsaaa" />
								<el-button @click="download1" v-if="yuchuliShow"
									style="margin-left: 10px;position: absolute;top: 10px;right: 10px;" size="small"
									type="primary">
									点击下载csv、excel文件</el-button>
							</div>
						</div>
					</div>

				</dv-border-box-13>
			</div>
			<div class="content_bottom">
				<div>
					<dv-border-box-13>
						<div class="sub_box">
							<div class="title">
								<el-select v-model="jiqi" placeholder="请选择">
									<el-option label="最小二乘回归" value="predict_pls">
									</el-option>
									<el-option label="knn回归" value="predict_knn">
									</el-option>
									<el-option label="xgboost回归" value="predict_xgboost">
									</el-option>
									<el-option label="gradientboost回归" value="predict_gradientboost">
									</el-option>
								</el-select>
								<el-upload class="upload-demo" :action="'http://82.156.0.244:59687/'+jiqi"
									:multiple="false" :file-list="fileList1" :before-upload="jiqiBefore"
									:on-success="jiqiSuccess" multiple :limit="1">
									<el-button style="margin-left: 10px;" size="small" type="primary">点击上传
										csv、excel文件</el-button>
								</el-upload>
							</div>
							<div class="list">
								<div v-for="(item,index) in ['CaCO3','K','N','OC','P','pH.in.H2O']" :key="item">
									<div class="texttitle">{{item}}</div>
									<div class="textnum" v-for="(e,i) in jiqiData[item]">{{e.toFixed(2)}}</div>
								</div>
							</div>
						</div>
					</dv-border-box-13>
				</div>
				<div>
					<dv-border-box-13>
						<div class="sub_box">
							<div class="title">
								<el-select v-model="shendu" placeholder="请选择">
									<el-option label="深度学习模型" value="1">
									</el-option>
								</el-select>
								<el-upload class="upload-demo" :action="'http://82.156.0.244:59687/predict_all'"
									:multiple="false" :file-list="fileList2" :before-upload="shenduBefore"
									:on-success="shenduSuccess" multiple :limit="1">
									<el-button style="margin-left: 10px;" size="small" type="primary">点击上传
										csv、excel文件</el-button>
								</el-upload>
							</div>
							<div class="list">
								<div v-for="(item,index) in ['caco3','k','n','oc','p','ph']" :key="item">
									<div class="texttitle">{{item}}</div>
									<div class="textnum" v-for="(e,i) in shenduData[item]">{{e.toFixed(2)}}</div>
								</div>
							</div>
						</div>
					</dv-border-box-13>
				</div>
			</div>
		</div>
		<div class="history">
			<dv-border-box-13>
				<div class="sub_box">
					<div>历史提交记录</div>
					<div class="history-list">
						<div v-for="(item,index) in history">
							<a :href="item.url" download>{{getTimeStr(item._add_time)}}</a>
						</div>
					</div>
				</div>
			</dv-border-box-13>
		</div>

	</div>
</template>

<script>
	import axios from 'axios';
	export default {
		data() {
			return {
				loading: false,
				fileList: [],
				yuchuliShow: false,
				yuchuli: "smooth",
				yuchuliOptionaaa: {
					grid: {
						bottom: '-10%',
					},
					xAxis: {
						type: 'category',
						data: [],
						splitLine: {
							show: false // 隐藏背景分割线
						}
					},
					yAxis: {
						type: 'value',
						splitLine: {
							show: false // 隐藏背景分割线
						}
					},
					series: []
				},
				yuchuliOptionbbb: {
					grid: {
						bottom: '-10%',
					},
					xAxis: {
						type: 'category',
						data: [],
						splitLine: {
							show: false // 隐藏背景分割线
						}
					},
					yAxis: {
						type: 'value',
						splitLine: {
							show: false // 隐藏背景分割线
						}
					},
					series: []
				},
				alldata: [],

				jiqi: "predict_pls",
				loading1: false,
				fileList1: [],
				jiqiData: {},

				shendu: "1",
				loading2: false,
				fileList2: [],
				shenduData: {},

				history: [],
			}
		},
		mounted() {
			this.getHistory();
		},
		methods: {

			getTimeStr(timestamp) {
				let date = new Date(timestamp);

				// 使用 Date 对象的方法获取年、月、日、时、分、秒等信息
				let year = date.getFullYear();
				let month = String(date.getMonth() + 1).padStart(2, '0'); // 月份从 0 开始计数，需要加1并补零
				let day = String(date.getDate()).padStart(2, '0');
				let hours = String(date.getHours()).padStart(2, '0');
				let minutes = String(date.getMinutes()).padStart(2, '0');
				let seconds = String(date.getSeconds()).padStart(2, '0');

				// 根据上述信息构建时间字符串
				return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
			},

			async getHistory() {
				axios.post(`https://fc-mp-8e18ce11-87b3-420d-9662-c5b9bfd2d497.next.bspapp.com/getHistory`).then(
					data => {
						this.history = data.data.data || data.data;
					})
			},

			async uploadSaveFile(file) {
				let that = this;
				const reader = new FileReader();
				reader.onload = function(event) {
					const base64String = event.target.result;
					axios.post(`https://fc-mp-8e18ce11-87b3-420d-9662-c5b9bfd2d497.next.bspapp.com/uploadFile`, {
						file: base64String
					}).then(
						data => {
							that.getHistory();
							console.log(data);
						})
				};

				// 开始读取文件并转换为Base64
				reader.readAsDataURL(file);
			},


			/**
			 * 预处理
			 */
			async yuchuliBefore(e) {
				this.yuchuliOptionaaa.series = [];
				this.yuchuliOptionbbb.series = [];
				this.loading = true;
				this.uploadSaveFile(e);
			},
			async yuchuliSuccess(e) {
				let data1 = [];
				for (let s of e.processed_data) {
					data1.push({
						data: Object.values(s),
						type: 'line',
						smooth: true
					});
				}
				this.yuchuliOptionaaa.series = data1;
				this.alldata = e.processed_data;

				let data2 = [];
				for (let s of e.original_data) {
					data2.push({
						data: Object.values(s),
						type: 'line',
						smooth: true
					});
				}
				this.yuchuliOptionbbb.series = data2;

				this.fileList = []; // 清空文件列表，允许再次上传
				this.yuchuliShow = true;
				this.loading = false;
			},
			download1() {
				let data = this.alldata;
				let templist = [];
				templist = Object.keys(data[0]).sort((a, b) => {
					return Number(a.split(".")[1]) - Number(b.split(".")[1]);
				});
				let str = templist.join(',') + "\n";
				//增加\t为了不让表格显示科学计数法或者其他格式
				for (let i = 0; i < data.length; i++) {
					for (let item in data[i]) {
						str += `${data[i][item]},`;
					}
					str += '\n';
				}

				let uri = 'data:text/csv;charset=utf-8,\ufeff' + encodeURIComponent(str);
				//通过创建a标签实现
				let link = document.createElement("a");
				link.href = uri;
				//对下载的文件命名
				link.download = "data.csv";
				document.body.appendChild(link);
				link.click();
				document.body.removeChild(link);
			},
			// async uploadYuchuli(e) {
			// 	const file = e.target.files[0];
			// 	const fd = new FormData();
			// 	fd.set("file", file);
			// 	axios.post(`http://82.156.0.244:59687/smooth`, fd).then(data => {
			// 		console.log(data);
			// 	})
			// },


			/**
			 * 机器学习
			 */
			async jiqiBefore(e) {
				console.log(e);
				this.loading1 = true;
				this.uploadSaveFile(e);
			},
			async jiqiSuccess(e) {
				this.jiqiData = e;
				this.fileList1 = []; // 清空文件列表，允许再次上传
				this.loading1 = false;
			},


			/**
			 * 深度学习
			 */
			async shenduBefore(e) {
				this.loading2 = true;
				this.uploadSaveFile(e);
			},
			async shenduSuccess(e) {
				this.shenduData = e;
				this.fileList2 = []; // 清空文件列表，允许再次上传
				this.loading2 = false;
			},

		},
	};
</script>

<style lang="scss" scoped>
	.box {
		padding: 30px 0;
		display: flex;
		height: 100%;
		flex: 1;

		.content {
			height: 90%;
			flex: 1;

			.content_top {
				height: 50%;
			}

			.content_bottom {
				height: 50%;
				display: flex;
				justify-content: space-between;

				&>div {
					width: 50%;
				}
			}
		}

		.history {
			height: 90%;
			width: 400px;
		}
	}

	.sub_box {
		padding: 30px;

		.title {
			display: flex;
		}
	}

	.echartsBox {
		height: 350px;
		display: flex;

		&>div {
			border: 1px dashed #666;
			width: 50%;
		}
	}

	.list {
		height: 350px;
		overflow-y: scroll;
		display: flex;
		justify-content: space-around;

		.texttitle {
			text-align: center;
			font-weight: bold;
		}

		.textnum {
			text-align: center;
			margin-top: 10px;
			font-size: 16px;
		}
	}

	.history-list {
		height: 800px;
		overflow-y: scroll;
		margin-top: 40px;

		&>div {
			margin-bottom: 10px;
		}
	}
</style>