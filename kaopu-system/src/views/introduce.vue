<template>
	<div class="index">
		<div class="bg_img">
			<img src="../assets/img/hero-bg.jpg" />
		</div>
		
	<!-- 	<div class="">
			<img src="../assets/img/hero-bg.jpg" />
		</div> -->
		
		<div class="projectTitle">靠 谱 kaopu</div>

		<div class="content">
			<div class="box">
				<div class="title">
					<div >基于近红外光谱和深度学习</div>
					<div style="margin-left: 150px;">的土壤养分含量预测系统</div>
				</div>
				<div class="sub_title">
					<div>【数据呈现】</div>
					可视化大屏，复杂数据变得简单直观。
				</div>
				<div class="sub_title">
					<div>【近红外光谱预处理】</div>
					对近红外光谱数据进行预处理，保证后续分析的准确性。
				</div>
				<div class="sub_title">
					<div>【机器学习回归分析】</div>
					利用机器学习的强大能力，快速构建预测模型。
				</div>
				<div class="sub_title">
					<div>【深度学习回归技术】</div>
					结合最新的深度学习技术，深入挖掘数据潜力。
				</div>
				<div class="btn">
					<div @click="jump('/index')">数据大屏</div>
					<div @click="jump('/algorithm')">养分预测</div>
				</div>
			</div>

		</div>
	</div>
</template>

<script>
	export default {
		methods: {
			jump(url) {
				this.$router.push(url);
			}
		},
	};
</script>

<style lang="scss" scoped>
	.index {
		width: 100vw;
		height: 100vh;
		position: relative;

		.bg_img {
			z-index: 99;
			position: absolute;
			width: 100%;
			height: 100%;
			overflow: hidden;

			&>img {
				width: auto;
				height: 160%;
			}
		}
		
		.projectTitle{
			top: 20px;
			left: 20px;
			position: absolute;
			z-index: 110;
			font-size: 2rem;
			color: #fff;
		}

		.content {
			position: relative;
			z-index: 100;
			width: 100%;
			height: 100%;
			background-color: rgba(0, 0, 0, 0.5);
			color: #fff;

			.box {
				position: absolute;
				right: 20%;
				top: 20%;

				.title {
					font-weight: bold;
					font-size: 3rem;

					&>div {
						line-height: 60px;
					}
					margin-bottom: 60px;
				}

				.sub_title {
					padding-left: 120px;
					// width: 500px;
					font-size: 1rem;
					margin: 30px 0;
					&>div{
						margin-bottom: 10px;
						font-weight: bold;
					}
				}

				.btn {
					padding-left: 120px;
					display: flex;

					&>div {
						cursor: pointer;
						color: #000;
						background-color: #fff;
						padding: 15px 40px;
						margin-right: 30px;

						&:last-child {
							color: #fff;
							background-color: #81b214;
						}
					}
				}
			}
		}
	}
</style>