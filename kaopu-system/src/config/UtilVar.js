/*
 * @Author: daidai
 * @Date: 2021-12-06 10:58:24
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2023-08-02 14:56:12
 * @FilePath: \web-pc\src\config\UtilVar.js
 */
let UtilVar = {
	ENC: false, //返回结果是否加密
	baseUrl: `http://82.156.0.244:59687`,
	tianqiUrl:"https://restapi.amap.com",
	code: 401,
}

const runtimeType = {
	production: () => {
		UtilVar.baseUrl = `http://82.156.0.244:59687`
		UtilVar.tianqiUrl = `https://restapi.amap.com`
	},
	//测试环境
	test: () => {

	},
	//开发环境
	development: () => {
		UtilVar.baseUrl = `/api`
		UtilVar.tianqiUrl = `/tianqi`
	},

}
console.log(process.env);

//通过打包配置打某个环境的api地址
runtimeType[process.env.VUE_APP_URL_ENV]()
export default UtilVar